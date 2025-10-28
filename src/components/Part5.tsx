import { Fab, Webchat } from "@botpress/webchat";
import { useCallback, useEffect, useState } from "react";
import { StylesheetProvider } from "@botpress/webchat";
// Import React to access React.CSSProperties
import React from 'react'; 

const config = {
  color: "#1cb02b",
  radius: 1,
  headerVariant: "solid",
};

const UseIsMobile = (breakpoint = 600)=>{
  const[isMobile,setIsMobile]=useState(false);

  const checkScreenSize=useCallback(()=>{
    if(typeof window !== 'undefined'){
      setIsMobile(window.innerWidth <= breakpoint);
    }
  },[breakpoint]);


  useEffect(() => {
    checkScreenSize();
    window.addEventListener('resize', checkScreenSize);
    return () => window.removeEventListener('resize', checkScreenSize);
  }, [checkScreenSize]);

  return isMobile;
}

function Part5() {
  const [storageKey] = useState(() => Date.now().toString())
  const [isWebchatOpen, setIsWebchatOpen] = useState(false);
  const isMobile = UseIsMobile(600);
  
  const toggleWebchat = () => {
    setIsWebchatOpen((prevState) => !prevState);
  };
  
  const baseWebchatStyle = {
    width: "400px",
    height: "500px",
    position: "fixed",
    bottom: "90px",
    right: "20px",
    zIndex: 9999,
    display: isWebchatOpen ? "flex" : "none",
  };
  
  // Define the responsive styles using the isMobile state
  const responsiveWebchatStyle = {
    ...baseWebchatStyle, 
    ...(isMobile && { 
      width: "70vw",
      height: "50vh",
      right: "2.5vw", 
      bottom: "10px", 
    }),
  };
  
  return (
    <>
      <Webchat
        clientId="39eecf5e-2abf-4645-933c-8ef3fb961752"
        configuration={{
          botName: "AI Waste Bot",
          botDescription: "Your eco-friendly assistant for waste management",
          botAvatar: "leaf.jpg",
        }}
        storageKey={storageKey}
        // APPLY THE TYPE ASSERTION HERE TO RESOLVE THE ERROR
        style={responsiveWebchatStyle as React.CSSProperties} 
      />
      <Fab
        onClick={() => toggleWebchat()}
        style={{
          position: "fixed",
          bottom: "20px",
          right: "20px",
          width: "64px",
          height: "64px",
          zIndex:9999
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

