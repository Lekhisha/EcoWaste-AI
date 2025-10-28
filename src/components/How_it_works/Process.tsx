
import process1 from "../../assests/process1.jpg";
import image4 from "../../assests/image4.jpg";
import image5 from "../../assests/image5.jpg";
const bgstyle2 = {
  backgroundImage: `url(${process1})`,
  height: "100%",
  width: "100%",
  backgroundPosition: "center",
  backgroundRepeat: "no-repeat",
  backgroundSize: "cover",
};
const bgstyle = {
  backgroundImage: `url(${image4})`,
  height: "100%",
  width: "100%",
  backgroundPosition: "center",
  backgroundRepeat: "no-repeat",
  backgroundSize: "cover",
};
const bgstyle3 = {
  backgroundImage: `url(${image5})`,
  height: "100%",
  width: "100%",
  backgroundPosition: "center",
  backgroundRepeat: "no-repeat",
  backgroundSize: "cover",
};
const Process = () => {
  return (
    <>
      <div className="relative py-14">
        <div className="container py-4">
          <div className="text-4xl font-bold text-cyan-950 text-center">
            Our Simple Process
          </div>
          <div className="pt-10">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 place-items-center">
              <div className="order-2 sm:order-1">
                <div className="text-2xl font-semibold text-cyan-950">
                  1.Capture or Upload Your Waste
                </div>
                <p className="text-cyan-950 font-semibold">
                  Effortlessly capture your waste item using your smartphone
                  camera or by uploading an existing image. Our intuitive
                  interface ensures a smooth and quick input process, ready for
                  identification.
                </p>
              </div>
              <div
                style={bgstyle2}
                className="rounded-3xl overflow-hidden  shadow-2xl order-1 sm:order-2"
              >
                <img src={process1} alt="" className="w-full" />
              </div>
            </div>
          </div>

           <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 place-items-center my-20">

             <div
                style={bgstyle}
                className="rounded-3xl overflow-hidden  shadow-2xl "
              >
                <img src={image4} alt="" className="w-full" />
              </div>

              <div className="">
                <div className="text-2xl font-semibold text-cyan-950">
                  2.AI Analyzes and Identifies
                </div>
                <p className="text-cyan-950 font-semibold">
                  Once captured, our advanced AI algorithms instantly process the image and data. It cross-references with a vast database to accurately identify the waste type, from common plastics to complex materials.
                </p>
              </div>
             
            </div>

             <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 place-items-center my-20">

              <div className="order-2 sm:order-1">
                <div className="text-2xl font-semibold text-cyan-950">
                  3.Instant,Clear Results
                </div>
                <p className="text-cyan-950 font-semibold">
                 Receive immediate, easy-to-understand identification results. Our system clearly displays the waste type, its recyclability status, and suggested disposal methods in a visually accessible layout.
                </p>
              </div>

              <div
                style={bgstyle3}
                className="rounded-3xl overflow-hidden  shadow-2xl order-1 sm:order-2"
              >
                <img src={image5} alt="" className="w-full" />
              </div>
             
            </div>
        </div>
      </div>
    </>
  );
};

export default Process;
