import React, { useState, useCallback, useEffect, useRef } from 'react';
import { Upload, Zap, Loader, AlertTriangle, Recycle, XCircle, Camera, X, SwitchCamera, Clock, Trash2 } from 'lucide-react';

// --- Type Definitions ---

// Type for the raw prediction returned by the Hugging Face model
interface HFPrediction {
    label: string;
    score: number;
}

// Type for the internal waste mapping structure
interface WasteDetail {
    type: string;
    recyclable: boolean;
    note: string;
    special?: boolean;
    compostable?: boolean;
}

// Type for the final classification result state
interface ClassificationResult extends WasteDetail {
    rawLabel: string;
}

// Type for a stored history item (includes the image data)
interface HistoryItem extends ClassificationResult {
    imageSrc: string; // Base64 data URL of the image
    timestamp: number;
}


// Union type for keys used in KEYWORD_SCORES and categoryScores
type WasteCategoryKey = 'PlasticFilm' | 'Metal' | 'RigidPlastic' | 'Paper' | 'Glass' | 'Cardboard' | 'FoodOrganics';

// Props for the RecyclabilityStatus component
interface RecyclabilityStatusProps {
    recyclable: boolean;
    special?: boolean;
    compostable?: boolean;
    note: string;
}

// --- Configuration ---

/**
 * The token is intentionally left blank here for security in the public editor.
 * TO RUN THIS APP HERE: Replace "" with your actual Hugging Face API token.
 * TO RUN THIS APP LOCALLY: Ensure VITE_HF_TOKEN is set in your .env file, and your local build system will inject it.
 */
const HF_API_TOKEN = import.meta.env.VITE_HF_TOKEN; 

// Using the general, fast model with advanced local scoring logic.
const API_URL = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224";

const FALLBACK_IMAGE_URL = "https://placehold.co/600x400/1e293b/f8fafc?text=Upload+Image";

// --- Utility Functions ---

/**
 * Converts a File or Blob object into a Base64 Data URL string.
 * @param file The File object to convert.
 */
const fileToBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result as string);
        reader.onerror = error => reject(error);
        reader.readAsDataURL(file);
    });
};

// --- Waste Classification Map and Scoring Configuration ---

// Global Keyword Scoring Configuration
const KEYWORD_SCORES: Record<WasteCategoryKey, string[]> = {
    'PlasticFilm': ['plastic bag', 'film', 'wrapping', 'wrap', 'grocery bag', 'carrier bag', 'plastic sheet', 'polyethylene', 'shopping bag', 'clear plastic'],
    'Metal': ['metal', 'steel', 'aluminum', 'aluminium', 'tin', 'can', 'foil', 'beverage can', 'utensil', 'cutlery', 'silverware', 'cookware'],
    'RigidPlastic': ['bottle', 'container', 'jug', 'rigid plastic', 'tub', 'pet bottle', 'plastic bottle'],
    'Paper': ['paper', 'book', 'magazine', 'newspaper', 'flyer', 'printed paper', 'notebook'],
    'Glass': ['glass', 'jar', 'shattered', 'cup', 'tumbler', 'cut glass', 'bottle', 'window'], 
    'Cardboard': ['box', 'cardboard', 'carton', 'packaging', 'paperboard', 'corrugated'],
    'FoodOrganics': ['banana', 'apple', 'orange', 'pizza slice', 'leaf', 'log', 'food', 'fruit', 'vegetable', 'compost', 'peel'],
};

const WASTE_MAP: Record<string, WasteDetail> = {
    'rigid_plastic_default': { type: 'Plastic (Rigid)', recyclable: true, note: 'Empty, rinse, and replace the cap. This is rigid plastic.' },
    'metal_default': { type: 'Metal', recyclable: true, note: 'Rinse well and flatten if possible.' },
    'glass_default': { type: 'Glass', recyclable: true, note: 'Empty and rinse well. **Intact** glass containers are recyclable.' },
    'cardboard_default': { type: 'Cardboard', recyclable: true, note: 'Must be flattened and dry. Remove all tape.' },
    'paper_default': { type: 'Paper', recyclable: true, note: 'Recyclable. Books, newspapers, and office paper should be clean and dry.' },
    
    'plastic_film_trash': { type: 'Miscellaneous Trash', recyclable: false, note: 'Plastic film/bags are NOT curbside recyclable. Use store drop-off or trash.' }, 
    'broken_glass_special': { type: 'Miscellaneous Trash', recyclable: false, special: true, 
        note: 'DANGER: Broken or shattered glass is NOT recyclable curbside due to safety. **Throw it in the trash** only after safely wrapping the pieces in thick newspaper or a small box.' 
    },
    'organic_compost': { type: 'Food Organics', recyclable: false, compostable: true, note: 'Compostable/Green Bin waste.' },
    
    'UNKNOWN_FALLBACK': { type: 'Miscellaneous Trash', recyclable: false, note: 'Item not in the specific waste map. Defaulting to Miscellaneous Trash.', compostable: false, special: false },
};

// --- API Utility ---

const retryFetch = async (url: string, options: RequestInit, retries: number = 3): Promise<Response> => {
    for (let i = 0; i < retries; i++) {
        try {
            const response = await fetch(url, options);
            if (!response.ok) {
                if (response.status === 401) {
                    throw new Error("401 Unauthorized: Please verify your Hugging Face API token is correct.");
                }
                const details = await response.json().catch(() => ({}));
                if (details.error && details.error.includes("is currently loading")) {
                    console.warn(`Model loading, retrying... (${i + 1}/${retries})`);
                    await new Promise(resolve => setTimeout(resolve, 5000)); 
                    continue; 
                }
                throw new Error(`HTTP error! Status: ${response.status}. Details: ${JSON.stringify(details).substring(0, 100)}...`);
            }
            return response;
        } catch (error: unknown) {
            const errorMessage = error instanceof Error ? error.message : "An unknown error occurred.";
            console.error(`Attempt ${i + 1} failed:`, errorMessage);
            if (i === retries - 1) throw new Error(errorMessage);
            await new Promise(resolve => setTimeout(resolve, Math.pow(2, i) * 1000));
        }
    }
    throw new Error('Fetch failed after all retries.'); 
};

// --- Core Logic: Multi-Label Scoring and Mapping ---
const getWasteDetails = (predictions: HFPrediction[]): ClassificationResult => {
    if (!predictions || predictions.length === 0) {
        return { rawLabel: 'N/A', ...WASTE_MAP['UNKNOWN_FALLBACK'] };
    }

    const categoryScores: Record<WasteCategoryKey, number> = {
        PlasticFilm: 0, RigidPlastic: 0, Metal: 0, Glass: 0, Cardboard: 0, Paper: 0, FoodOrganics: 0,
    };
    
    let topRawLabel = predictions[0].label;
    
    predictions.slice(0, 5).forEach((p: HFPrediction) => {
        const cleanLabel = p.label.toLowerCase().trim();
        const weight = p.score; 

        for (const category in KEYWORD_SCORES) {
            if (KEYWORD_SCORES.hasOwnProperty(category)) {
                const key = category as WasteCategoryKey;
                KEYWORD_SCORES[key].forEach((keyword: string) => {
                    if (cleanLabel.includes(keyword)) {
                        categoryScores[key] += (weight + 0.1); 
                    }
                });
            }
        }
    });

    let winningCategory: WasteCategoryKey | null = null;
    let maxScore = 0;

    for (const category in categoryScores) {
        if (categoryScores.hasOwnProperty(category)) {
            const key = category as WasteCategoryKey;
            if (categoryScores[key] > maxScore && categoryScores[key] > 0.3) {
                maxScore = categoryScores[key];
                winningCategory = key;
            }
        }
    }
    
    const lowerTopLabel = topRawLabel.toLowerCase();

    // CRITICAL: Prioritize BROKEN GLASS/SHATTERED check first for safety!
    if (lowerTopLabel.includes('broken glass') || lowerTopLabel.includes('shattered') || lowerTopLabel.includes('broken cup') || lowerTopLabel.includes('cut glass')) {
        return { rawLabel: topRawLabel, ...WASTE_MAP['broken_glass_special'] };
    }

    switch (winningCategory) {
        case 'RigidPlastic': return { rawLabel: topRawLabel, ...WASTE_MAP['rigid_plastic_default'] };
        case 'PlasticFilm': return { rawLabel: topRawLabel, ...WASTE_MAP['plastic_film_trash'] };
        case 'Metal': return { rawLabel: topRawLabel, ...WASTE_MAP['metal_default'] };
        case 'Cardboard': return { rawLabel: topRawLabel, ...WASTE_MAP['cardboard_default'] };
        case 'Paper': return { rawLabel: topRawLabel, ...WASTE_MAP['paper_default'] };
        case 'Glass': return { rawLabel: topRawLabel, ...WASTE_MAP['glass_default'] };
        case 'FoodOrganics': return { rawLabel: topRawLabel, ...WASTE_MAP['organic_compost'] };
        default:
            if (lowerTopLabel.includes('bottle') || lowerTopLabel.includes('can') || lowerTopLabel.includes('book')) {
                if (lowerTopLabel.includes('book') || lowerTopLabel.includes('newspaper')) {
                    return { rawLabel: topRawLabel, ...WASTE_MAP['paper_default'], note: `Model was vague but identified a paper product. Assuming clean and dry paper.` };
                }
                return { rawLabel: topRawLabel, ...WASTE_MAP['rigid_plastic_default'], note: `Model was vague but identified a container type. Assuming standard rigid plastic.` };
            }
            return { rawLabel: topRawLabel, ...WASTE_MAP['UNKNOWN_FALLBACK'] };
    }
};


// --- Main React Component ---

const Part2: React.FC = () => {
    // State
    const [imageFile, setImageFile] = useState<File | null>(null);
    const [imageUrl, setImageUrl] = useState<string>(FALLBACK_IMAGE_URL);
    const [classification, setClassification] = useState<ClassificationResult | null>(null); 
    const [loading, setLoading] = useState<boolean>(false);
    const [error, setError] = useState<string>('');
    const [isCameraMode, setIsCameraMode] = useState<boolean>(false);
    // State for history, limited to 5 items
    const [history, setHistory] = useState<HistoryItem[]>([]);
    
    // Camera States for multi-camera support
    const [cameraFacingMode, setCameraFacingMode] = useState<'user' | 'environment'>('environment');
    const [hasMultipleCameras, setHasMultipleCameras] = useState<boolean>(false);

    // Refs for camera and canvas
    const videoRef = useRef<HTMLVideoElement>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null); // Ref for direct input manipulation

    // --- Camera Control Logic ---
    
    const stopCamera = useCallback(() => {
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
            streamRef.current = null;
        }
    }, []);

    // Function to start the stream based on a given mode
    const startStream = useCallback(async (mode: 'user' | 'environment') => {
        setError('');
        
        // Stop any existing stream first
        stopCamera(); 

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    facingMode: mode 
                } 
            });
            streamRef.current = stream;
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                videoRef.current.play();
                setIsCameraMode(true);
                // Clear the preview image when camera starts
                setImageUrl(FALLBACK_IMAGE_URL); 
            }
        } catch (e) {
            console.error("Camera access failed:", e);
            // If environment mode fails, try user mode as fallback if not explicitly requested
            if (mode === 'environment' && cameraFacingMode !== 'user') {
                console.warn("Environment camera failed, attempting user camera.");
                startStream('user');
                setCameraFacingMode('user');
                return;
            }
            setError(`Could not access camera (${mode} mode). Please ensure permissions are granted.`);
            setIsCameraMode(false); // Make sure mode is off on failure
        }
    }, [stopCamera, cameraFacingMode]);

    // Effect to manage the camera stream lifecycle based on state changes
    useEffect(() => {
        if (isCameraMode) {
            startStream(cameraFacingMode);
        } else {
            stopCamera();
        }
    }, [isCameraMode, cameraFacingMode, startStream, stopCamera]); 

    // Effect to check for multiple cameras on mount
    useEffect(() => {
        const checkCameras = async () => {
            if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
            try {
                const devices = await navigator.mediaDevices.enumerateDevices();
                const videoInputs = devices.filter(device => device.kind === 'videoinput');
                setHasMultipleCameras(videoInputs.length > 1);
            } catch (e) {
                console.error("Error enumerating devices:", e);
            }
        };
        checkCameras();
    }, []); 

    // Toggles the facing mode (which triggers the useEffect to restart the stream)
    const toggleCameraFacingMode = useCallback(() => {
        // Toggle the state. useEffect will handle stopping and restarting the stream.
        setCameraFacingMode(prevMode => (prevMode === 'environment' ? 'user' : 'environment'));
    }, []);
    
    // Function called by the UI button to activate camera mode
    const handleStartCameraMode = () => {
        setError('');
        setImageFile(null);
        setClassification(null);
        setIsCameraMode(true);
    };

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files ? event.target.files[0] : null;

        // Reset the input value here after selection, which is safer than using onClick
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }

        if (isCameraMode) {
             stopCamera();
             setIsCameraMode(false);
        }

        if (file && file.type.startsWith('image/')) {
            setImageFile(file);
            setImageUrl(URL.createObjectURL(file));
            setClassification(null);
            setError('');
        } else {
            setImageFile(null);
            setImageUrl(FALLBACK_IMAGE_URL);
            setError('Please select a valid image file (PNG, JPG, etc.).');
        }
    };
    
    // Function to capture the image from the video stream
    const captureImage = useCallback((): Promise<File | null> => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (!video || !canvas) {
            setError("Camera or canvas not ready for capture.");
            return Promise.resolve(null);
        }

        // Set canvas dimensions to match video stream
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const ctx = canvas.getContext('2d');
        if (!ctx) return Promise.resolve(null);

        // Apply mirror transformation if using the user-facing camera (optional for capture, but cleaner)
        if (cameraFacingMode === 'user') {
            ctx.translate(canvas.width, 0);
            ctx.scale(-1, 1);
        }
        
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Reset transformation for future use
        if (cameraFacingMode === 'user') {
            ctx.setTransform(1, 0, 0, 0, 0, 0);
        }

        return new Promise((resolve) => {
            canvas.toBlob((blob) => {
                if (blob) {
                    const file = new File([blob], 'captured_waste_image.png', { type: 'image/png' });
                    setImageFile(file);
                    setImageUrl(URL.createObjectURL(file));
                    resolve(file);
                } else {
                    setError("Failed to create image blob from canvas.");
                    resolve(null);
                }
            }, 'image/png');
        });
    }, [cameraFacingMode]);


    const runInference = useCallback(async () => {
        // Token Check
        if (HF_API_TOKEN === "") { 
             setError('Hugging Face API Token is missing. If running locally, ensure VITE_HF_TOKEN is set in your .env file. If running here, you must manually update the HF_API_TOKEN constant in the source code.');
             return;
        }

        let fileToInfer: File | null = imageFile;

        if (isCameraMode) {
            fileToInfer = await captureImage();
            setIsCameraMode(false); // Exiting camera mode after successful capture or failure
            if (!fileToInfer) {
                setLoading(false);
                return; // Capture failed, error already set in captureImage
            }
        } else if (!imageFile) {
            setError('Please upload an image or enable camera mode before classifying.');
            return;
        }

        if (!fileToInfer) { 
             setError('No image available for inference.');
             return;
        }
        
        setLoading(true);
        setError('');
        setClassification(null);

        try {
            // 1. Convert File to Base64 Data URL for history BEFORE inference starts
            const base64Image = await fileToBase64(fileToInfer);

            const imageBuffer = await fileToInfer.arrayBuffer();

            const options: RequestInit = {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${HF_API_TOKEN}`,
                    'Content-Type': fileToInfer.type,
                },
                body: imageBuffer,
            };

            const response = await retryFetch(API_URL, options);
            const result: HFPrediction[] | { error: string } = await response.json();
            
            if (Array.isArray(result) && result.length > 0) {
                const wasteDetails = getWasteDetails(result);
                setClassification(wasteDetails);

                // 2. Add to history (FIFO, max 5)
                const newHistoryItem: HistoryItem = { 
                    ...wasteDetails, 
                    imageSrc: base64Image, 
                    timestamp: Date.now() 
                };
                
                setHistory(prevHistory => {
                    const updatedHistory = [newHistoryItem, ...prevHistory];
                    // Keep only the latest 5 entries
                    return updatedHistory.slice(0, 5); 
                });


            } else if (!Array.isArray(result) && 'error' in result) {
                 setError(`Classification failed: Hugging Face API Error: ${result.error}`);
            } else {
                setError('Classification failed: Received an unexpected response format from the model.');
            }

        } catch (err: unknown) {
            const errorMessage = err instanceof Error ? err.message : "An unknown error occurred during inference.";
            console.error("Inference Error:", err);
            setError(`Inference failed: ${errorMessage}`);
        } finally {
            setLoading(false);
        }
    }, [imageFile, isCameraMode, captureImage]); 

    // Cleanup function for object URL and camera stream
    useEffect(() => {
        return () => {
            if (imageUrl && imageUrl !== FALLBACK_IMAGE_URL) {
                URL.revokeObjectURL(imageUrl);
            }
            stopCamera();
        };
    }, [imageUrl, stopCamera]); 

    // Component to render the Recyclability status
    const RecyclabilityStatus: React.FC<RecyclabilityStatusProps> = ({ recyclable, special, compostable, note }) => {
        let icon, color, text;

        if (special) {
            icon = <AlertTriangle className="w-6 h-6" />;
            color = 'text-red-600 bg-red-100 dark:bg-red-800 dark:text-red-200';
            text = 'Special Disposal Required';
        } else if (recyclable) {
            icon = <Recycle className="w-6 h-6" />;
            color = 'text-green-600 bg-green-100 dark:bg-green-800 dark:text-green-200';
            text = 'Recyclable';
        } else if (compostable) {
            icon = <Zap className="w-6 h-6" />;
            color = 'text-yellow-600 bg-yellow-100 dark:bg-yellow-800 dark:text-yellow-200';
            text = 'Compostable / Green Bin';
        } else {
            icon = <XCircle className="w-6 h-6" />;
            color = 'text-gray-600 bg-gray-100 dark:bg-gray-700 dark:text-gray-200';
            text = 'General Waste (Not Recyclable)';
        }

        return (
            <div className={`p-4 rounded-xl flex items-center transition duration-200 ${color}`}>
                <div className="mr-4 shrink-0">{icon}</div>
                <div>
                    <h3 className="text-xl font-bold">{text}</h3>
                    <p className="text-sm font-medium mt-1">{note}</p>
                </div>
            </div>
        );
    };

    // History Item Component
    const HistoryCard: React.FC<{ item: HistoryItem }> = ({ item }) => {
        const getIndicatorColor = () => {
            if (item.special) return 'bg-red-500';
            if (item.recyclable) return 'bg-green-500';
            if (item.compostable) return 'bg-yellow-500';
            return 'bg-gray-500';
        };

        return (
            <div className="bg-white dark:bg-gray-700 rounded-lg shadow-lg overflow-hidden transition hover:shadow-xl duration-300">
                <div className="relative h-32 w-full">
                    <img
                        src={item.imageSrc}
                        alt="Classified Item"
                        className="w-full h-full object-cover"
                        loading="lazy"
                    />
                    <div className={`absolute top-0 right-0 p-2 rounded-bl-lg text-white font-bold text-xs ${getIndicatorColor()}`}>
                        {item.recyclable ? 'RECYCLE' : item.compostable ? 'COMPOST' : 'TRASH'}
                    </div>
                </div>
                <div className="p-3">
                    <p className="text-sm font-bold text-gray-800 dark:text-gray-100 truncate">{item.type}</p>
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-1 flex items-center">
                        <Clock className="w-3 h-3 mr-1" />
                        {new Date(item.timestamp).toLocaleTimeString()}
                    </p>
                    <p className="text-xs mt-2 italic text-gray-600 dark:text-gray-300 line-clamp-2">
                        {item.note}
                    </p>
                </div>
            </div>
        );
    };


    return (
        <div className="min-h-screen bg-gray-50 dark:bg-gray-900 p-4 sm:p-8 font-inter">
            {/* Load Tailwind CSS CDN */}
            <script src="https://cdn.tailwindcss.com"></script>
            {/* Custom Styles - only use standard CSS here. */}
            <style>{`
                .font-inter { font-family: 'Inter', sans-serif; }
                .shadow-3xl { box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04); }
                .camera-video {
                    object-fit: cover;
                    /* Mirror only front camera, keep environment camera normal */
                    transform: ${cameraFacingMode === 'user' ? 'scaleX(-1)' : 'none'}; 
                }
            `}</style>

            <div className="max-w-4xl mx-auto">
                <header className="text-center mb-10">
                    <h1 className="text-4xl text-cyan-950 dark:text-cyan-50 font-extrabold">
                        AI Waste Classifier
                    </h1>
                    <p className="text-gray-500 dark:text-gray-400 mt-2">
                        Identify items using computer vision to get instant disposal advice.
                    </p>
                </header>

                <div className="bg-white dark:bg-gray-800 rounded-xl shadow-3xl p-6 md:p-10 border border-gray-100 dark:border-gray-700">
                    
                    {/* Mode Toggle Button Row */}
                    <div className="flex justify-center space-x-4 mb-8">
                        
                        {/* Only visible when camera is ON and multiple cams exist */}
                        {isCameraMode && hasMultipleCameras && (
                            <button
                                onClick={toggleCameraFacingMode}
                                className="flex items-center px-6 py-3 text-base font-bold text-white bg-indigo-500 rounded-full hover:bg-indigo-600 transition shadow-lg"
                                title={cameraFacingMode === 'environment' ? "Switch to Front Camera" : "Switch to Back Camera"}
                            >
                                <SwitchCamera className="w-5 h-5 mr-2" /> 
                                {cameraFacingMode === 'environment' ? 'Front Cam' : 'Back Cam'}
                            </button>
                        )}

                        {!isCameraMode ? (
                            <button
                                onClick={handleStartCameraMode}
                                className="flex items-center px-6 py-3 text-base font-bold text-white bg-green-500 rounded-full hover:bg-green-600 transition shadow-lg"
                            >
                                <Camera className="w-5 h-5 mr-2" /> Use Camera
                            </button>
                        ) : (
                            <button
                                onClick={() => setIsCameraMode(false)}
                                className="flex items-center px-6 py-3 text-base font-bold text-white bg-red-500 rounded-full hover:bg-red-600 transition shadow-lg"
                            >
                                <X className="w-5 h-5 mr-2" /> Close Camera
                            </button>
                        )}
                    </div>
                    
                    {/* Image Input / Camera View Section */}
                    <div className="flex flex-col md:flex-row gap-8 mb-8">
                        
                        {/* Input/Camera Area */}
                        <div className="flex-1 min-h-[250px] flex items-center justify-center">
                            {!isCameraMode ? (
                                // File Upload Mode
                                <label className="cursor-pointer w-full border-4 border-dashed border-green-400 dark:border-green-600 p-6 rounded-xl hover:border-green-700 transition duration-300 relative bg-indigo-50 dark:bg-gray-700 h-full flex flex-col justify-center items-center">
                                    <input
                                        ref={fileInputRef} // Added ref here
                                        type="file"
                                        accept="image/*"
                                        // FIX APPLIED: Removed the 'capture' attribute AND the interfering 'onClick' handler.
                                        onChange={handleFileChange}
                                        className="hidden"
                                    />
                                    <Upload className="w-10 h-10 text-green-500 mb-3" />
                                    <p className="font-semibold text-green-600 dark:text-green-300 text-center">
                                        {imageFile ? `Change File: ${imageFile.name}` : "Click to Upload Waste Image"}
                                    </p>
                                </label>
                            ) : (
                                // Camera Mode
                                <div className="w-full h-full rounded-xl overflow-hidden shadow-xl border border-gray-200 dark:border-gray-600">
                                    <video
                                        ref={videoRef}
                                        className="camera-video w-full h-full min-h-[300px]"
                                        autoPlay
                                        playsInline
                                    />
                                </div>
                            )}
                        </div>

                        {/* Image Preview */}
                        <div className="flex-1 w-full max-w-sm mx-auto">
                            <img
                                src={imageUrl}
                                alt="Image Preview"
                                className="w-full object-cover rounded-xl border border-gray-200 dark:border-gray-600 shadow-md min-h-[250px]"
                                onError={(e) => {
                                    const target = e.target as HTMLImageElement;
                                    target.src = FALLBACK_IMAGE_URL;
                                }}
                            />
                        </div>
                    </div>
                    
                    {/* Hidden Canvas for capture (necessary for toBlob) */}
                    <canvas ref={canvasRef} style={{ display: 'none' }} />


                    {/* Action Button (Analyze) */}
                    <div className="flex justify-center mb-8">
                        <button
                            onClick={runInference}
                            disabled={loading || (!imageFile && !isCameraMode)}
                            className={`px-10 py-3 rounded-full font-bold text-lg transition duration-300 shadow-xl flex items-center justify-center ${
                                loading || (!imageFile && !isCameraMode)
                                    ? 'bg-green-400 cursor-not-allowed text-white'
                                    : 'bg-green-600 hover:bg-green-700 active:ring-4 ring-green-300 text-white'
                            }`}
                        >
                            {loading ? (
                                <><Loader className="w-5 h-5 mr-3 animate-spin" /> Classifying...</>
                            ) : isCameraMode ? (
                                <><Camera className="w-5 h-5 mr-2" /> Capture & Analyze</>
                            ) : (
                                <> Analyze Uploaded Item</>
                            )}
                        </button>
                    </div>

                    {/* Error Display */}
                    {error && (
                        <div className="mt-6 p-4 bg-red-100 dark:bg-red-900 border border-red-400 dark:border-red-700 text-red-700 dark:text-red-300 rounded-lg flex items-start">
                            <AlertTriangle className="w-5 h-5 mr-3 mt-0.5 shrink-0" />
                            <div>
                                <h3 className="font-bold">Error</h3>
                                <p className="text-sm">{error}</p>
                            </div>
                        </div>
                    )}

                    {/* Results Display */}
                    {classification && !loading && (
                        <div className="mt-8 space-y-6">
                            <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-100 mb-4 border-b pb-3 border-gray-200 dark:border-gray-700">
                                Classification Details
                            </h2>
                            
                            {/* Waste Type & Model Label */}
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <div className="p-4 bg-blue-50 dark:bg-blue-900 rounded-lg border border-blue-200 dark:border-blue-700">
                                    <p className="text-sm font-medium text-blue-600 dark:text-blue-300">Waste Type (Scored)</p>
                                    <p className="text-2xl font-extrabold text-blue-800 dark:text-blue-100">
                                        {classification.type}
                                    </p>
                                </div>
                                <div className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg border border-gray-200 dark:border-gray-600">
                                    <p className="text-sm font-medium text-gray-600 dark:text-gray-300">Raw Model Label (Confidence Top 1)</p>
                                    <p className="text-lg font-bold text-gray-800 dark:text-gray-100 italic">
                                        {classification.rawLabel}
                                    </p>
                                </div>
                            </div>
                            
                            {/* Recyclability Status Component */}
                            <RecyclabilityStatus 
                                recyclable={classification.recyclable}
                                special={classification.special}
                                compostable={classification.compostable}
                                note={classification.note}
                            />

                        </div>
                    )}
                    
                    {/* History Section */}
                    {history.length > 0 && (
                        <div className="mt-12 pt-8 border-t border-gray-200 dark:border-gray-700">
                            <div className="flex justify-between items-center mb-6">
                                <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-100">
                                    Recent Classifications ({history.length})
                                </h2>
                                <button
                                    onClick={() => setHistory([])}
                                    className="text-sm font-medium text-red-500 hover:text-red-700 dark:text-red-400 dark:hover:text-red-300 transition flex items-center"
                                >
                                    <Trash2 className="w-4 h-4 mr-1" /> Clear History
                                </button>
                            </div>
                            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
                                {history.map((item, index) => (
                                    <HistoryCard key={item.timestamp + index} item={item} />
                                ))}
                            </div>
                        </div>
                    )}
                
                    {/* End of main card */}
                </div>
            </div>
        </div>
    );
};

export default Part2;








