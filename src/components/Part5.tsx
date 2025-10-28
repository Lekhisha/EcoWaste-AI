import { Fab, Webchat, StylesheetProvider } from "@botpress/webchat";
import { useCallback, useEffect, useState, type CSSProperties } from "react";

// --- Configuration ---
const config = {
  color: "#1cb02b",
  radius: 1,
  headerVariant: "solid",
};

// --- Custom Hook to Check Mobile Size ---
const useIsMobile = (breakpoint = 600) => {
  const [isMobile, setIsMobile] = useState(false);

  const checkScreenSize = useCallback(() => {
    if (typeof window !== "undefined") {
      setIsMobile(window.innerWidth <= breakpoint);
    }
  }, [breakpoint]);

  useEffect(() => {
    checkScreenSize();
    window.addEventListener("resize", checkScreenSize);
    return () => window.removeEventListener("resize", checkScreenSize);
  }, [checkScreenSize]);

  return isMobile;
};

// --- Webchat Wrapper (filters invalid DOM props) ---
interface WebchatWrapperProps {
  clientId: string;
  configuration: any;
  storageKey: string;
  style: CSSProperties;
}

function WebchatWrapper({
  clientId,
  configuration,
  storageKey,
  style,
}: WebchatWrapperProps) {
  return (
    <Webchat
      clientId={clientId}
      configuration={configuration}
      storageKey={storageKey}
      style={style}
    />
  );
}

// --- Main Component ---
function Part5() {
  const [storageKey] = useState(() => Date.now().toString());
  const [isWebchatOpen, setIsWebchatOpen] = useState(false);
  const isMobile = useIsMobile(600);

  const toggleWebchat = () => {
    setIsWebchatOpen((prev) => !prev);
  };

  const baseWebchatStyle: CSSProperties = {
    width: "400px",
    height: "500px",
    position: "fixed",
    bottom: "90px",
    right: "20px",
    zIndex: 9999,
    display: isWebchatOpen ? "flex" : "none",
    flexDirection: "column",
  };

  const responsiveWebchatStyle: CSSProperties = {
    ...baseWebchatStyle,
    ...(isMobile && {
      width: "90vw",
      height: "80vh",
      right: "5vw",
      bottom: "10px",
    }),
  };

  const webchatConfiguration = {
    clientId: "39eecf5e-2abf-4645-933c-8ef3fb961752",
    botName: "AI Waste Bot",
    botDescription: "Your eco-friendly assistant for waste management",
    botAvatar: "/leaf.jpg",
  };

  return (
    <>
      <WebchatWrapper
        clientId={webchatConfiguration.clientId}
        configuration={webchatConfiguration}
        storageKey={storageKey}
        style={responsiveWebchatStyle}
      />

      <Fab
        onClick={toggleWebchat}
        style={{
          position: "fixed",
          bottom: "20px",
          right: "20px",
          width: "64px",
          height: "64px",
          zIndex: 9999,
        }}
      />

      <StylesheetProvider
        color={config.color}
        fontFamily="inter"
        radius={config.radius}
        variant="soft"
        themeMode="light"
      />
    </>
  );
}

export default Part5;

