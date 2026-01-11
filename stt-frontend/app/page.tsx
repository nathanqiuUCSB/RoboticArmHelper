"use client";

import { useState, useRef, useEffect } from "react";

interface Message {
  type: "user" | "robot";
  text: string;
}

async function sendRobotCommand(text: string) {
  const res = await fetch(`http://localhost:8000/robot/command/${text}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  }); 

  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`Robot command failed (${res.status}): ${errText}`);
  }

  return res.json(); // {status: "started"} (or whatever you return)
}


export default function Home() {
  const [recording, setRecording] = useState(false);
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isScrolledUp, setIsScrolledUp] = useState(false);
  const [isScrolling, setIsScrolling] = useState(false);
  const scrollTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunks = useRef<Blob[]>([]);
  const videoRef = useRef<HTMLVideoElement>(null);
  const consoleRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (consoleRef.current) {
      consoleRef.current.scrollTop = consoleRef.current.scrollHeight;
      setIsScrolledUp(false);
      setIsScrolling(false); // Hide scrollbar when auto-scrolling
    }
  }, [messages]);
  
  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (scrollTimeoutRef.current) {
        clearTimeout(scrollTimeoutRef.current);
      }
    };
  }, []);

  // Handle scroll detection
  const handleScroll = () => {
    if (consoleRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = consoleRef.current;
      const isAtBottom = scrollHeight - scrollTop - clientHeight < 10;
      setIsScrolledUp(!isAtBottom);
      setIsScrolling(true);
      
      // Clear existing timeout
      if (scrollTimeoutRef.current) {
        clearTimeout(scrollTimeoutRef.current);
      }
      
      // Hide scrollbar after scrolling stops (300ms delay)
      scrollTimeoutRef.current = setTimeout(() => {
        setIsScrolling(false);
      }, 300);
    }
  };

  // Initialize camera
  useEffect(() => {
    const initCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
          video: { width: 640, height: 480 },
          audio: false 
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (error) {
        console.error("Error accessing camera:", error);
      }
    };

    initCamera();
    return () => {
      if (videoRef.current?.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  async function startRecording() {
    setText("");
    setLoading(false);
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    const mediaRecorder = new MediaRecorder(stream);
    mediaRecorderRef.current = mediaRecorder;
    audioChunks.current = [];

    mediaRecorder.ondataavailable = (event) => {
      audioChunks.current.push(event.data);
    };

    mediaRecorder.onstop = async () => {
      // Only process if we have meaningful audio data (at least 100ms)
      const audioBlob = new Blob(audioChunks.current, { type: mediaRecorder.mimeType || "audio/webm" });
      if (audioBlob.size > 1000) { // At least 1KB of audio data
        setLoading(true);
        await sendAudio(audioBlob);
      } else {
        setLoading(false);
      }
    };

    mediaRecorder.start();
    setRecording(true);
  }

  function stopRecording() {
    mediaRecorderRef.current?.stop();
    setRecording(false);
  }

  async function sendAudio(blob: Blob) {
    try {
      const formData = new FormData();
      const extension = blob.type.includes("webm") ? ".webm" : blob.type.includes("ogg") ? ".ogg" : ".wav";
      formData.append("audio", blob, `recording${extension}`);

      const res = await fetch("http://localhost:8000/transcribe", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`Server error: ${res.status}`);
      }

      const data = await res.json();
      const userText = data.text;
      setText(userText);
      
      // Only add robot response, not user message
      // Filter out error messages and invalid text
      if (userText && 
          userText !== "Could not understand audio" && 
          !userText.startsWith("Error:") &&
          !userText.includes("--enable") &&
          userText.length < 500) { // Filter out suspiciously long error messages
        // Add robot response (for now, just echo back or process it)
        // TODO: Replace with actual robot response logic
        //setTimeout(() => {
         // setMessages(prev => [...prev, { type: "robot", text: `Understood: ${userText}` }]);
       // }, 500);
       try{
        await sendRobotCommand(userText);
       } catch (err){
        console.log("Failed to send robot command")
       }
      }
    } catch (error) {
      console.error("Error sending audio:", error);
      const errorMsg = `Error: ${error instanceof Error ? error.message : "Failed to transcribe audio"}`;
      setText(errorMsg);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main style={{
      height: "100vh",
      background: "white",
      display: "flex",
      flexDirection: "column",
      padding: "1rem",
      fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif",
      overflow: "hidden",
    }}>
      {/* Header */}
      <div style={{
        textAlign: "center",
        color: "#111827",
        marginBottom: "0.75rem",
        flexShrink: 0,
      }}>
        <h1 style={{
          fontSize: "2.5rem",
          fontWeight: "700",
          margin: "0",
        }}>
          ExtendAble
        </h1>
      </div>

      {/* Main Camera Feed */}
      <div style={{
        flex: 1,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        maxWidth: "800px",
        width: "100%",
        margin: "0 auto",
        minHeight: 0,
      }}>
        <div style={{
          background: "white",
          borderRadius: "16px",
          boxShadow: "0 20px 60px rgba(0, 0, 0, 0.3)",
          padding: "1rem",
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: "0.75rem",
          minHeight: 0,
        }}>
          {/* Camera Feed */}
          <div style={{
            width: "100%",
            flex: 1,
            borderRadius: "12px",
            overflow: "hidden",
            background: "#000",
            position: "relative",
            minHeight: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              style={{
                width: "100%",
                height: "100%",
                objectFit: "contain",
              }}
            />
          </div>

          {/* Controls Row - Button and Logs */}
          <div style={{
            display: "flex",
            alignItems: "center",
            gap: "1rem",
            width: "100%",
            justifyContent: "space-between",
            flexShrink: 0,
          }}>
            {/* Push to Talk Button */}
            <button
              onMouseDown={startRecording}
              onMouseUp={stopRecording}
              onTouchStart={(e) => {
                e.preventDefault();
                startRecording();
              }}
              onTouchEnd={(e) => {
                e.preventDefault();
                stopRecording();
              }}
              disabled={loading}
              style={{
                width: "60px",
                height: "60px",
                borderRadius: "50%",
                border: "none",
                background: recording
                  ? "linear-gradient(135deg, #ef4444 0%, #dc2626 100%)"
                  : "linear-gradient(135deg,rgb(255, 255, 255) 0%, #764ba2 100%)",
                color: "white",
                fontSize: "1.25rem",
                fontWeight: "600",
                cursor: loading ? "not-allowed" : "pointer",
                boxShadow: recording
                  ? "0 0 15px rgba(239, 68, 68, 0.5)"
                  : "0 4px 12px rgba(102, 126, 234, 0.3)",
                transition: "all 0.3s ease",
                transform: recording ? "scale(1.05)" : "scale(1)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                outline: "none",
                flexShrink: 0,
              }}
              onMouseLeave={() => {
                if (recording) stopRecording();
              }}
            >
              {loading ? (
                <span style={{
                  width: "20px",
                  height: "20px",
                  border: "2px solid rgba(255,255,255,0.3)",
                  borderTop: "2px solid white",
                  borderRadius: "50%",
                  animation: "spin 1s linear infinite",
                  display: "inline-block",
                }} />
              ) : recording ? (
                "🎙️"
              ) : (
                "🎤"
              )}
            </button>

            {/* Conversation Display */}
            <div 
              ref={consoleRef}
              onScroll={handleScroll}
              className={`conversation-container ${isScrolling ? "scrolling" : ""}`}
              style={{
                flex: 1,
                background: "white",
                borderRadius: "8px",
                padding: "0.75rem",
                border: "1px solid #e5e7eb",
                textAlign: "left",
                height: "60px",
                overflowY: "auto",
                scrollbarWidth: isScrolling ? "thin" : "none",
                scrollbarColor: isScrolling ? "#cbd5e1 transparent" : "transparent transparent",
                scrollSnapType: "y mandatory",
                position: "relative",
                transition: "scrollbar-width 0.3s ease",
              }}
            >
              {recording ? (
                <div style={{ 
                  color: "#9ca3af", 
                  fontStyle: "italic", 
                  fontSize: "1.125rem",
                  height: "60px",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  position: "absolute",
                  width: "100%",
                  padding: "0 0.75rem",
                }}>
                  Listening...
                </div>
              ) : messages.filter(msg => msg.type === "robot").length === 0 ? (
                <div style={{ 
                  color: "#9ca3af", 
                  fontStyle: "italic", 
                  fontSize: "1.125rem",
                  height: "60px",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  position: "absolute",
                  width: "100%",
                  padding: "0 0.75rem",
                }}>
                  Start a conversation...
                </div>
              ) : (
                <>
                  {messages.filter(msg => msg.type === "robot").map((msg, index, arr) => (
                    <div
                      key={index}
                      style={{
                        fontSize: "1.125rem",
                        lineHeight: "1.6",
                        wordBreak: "break-word",
                        height: "60px",
                        display: "flex",
                        alignItems: "center",
                        scrollSnapAlign: "start",
                        padding: "0 0.75rem",
                        boxSizing: "border-box",
                      }}
                    >
                      <span style={{ 
                        fontWeight: "600",
                        color: "#7c3aed",
                        marginRight: "0.5rem",
                      }}>
                        ExtendAble:
                      </span>
                      <span style={{ color: "#111827" }}>
                        {msg.text}
                      </span>
                    </div>
                  ))}
                </>
              )}
            </div>
          </div>

    
        </div>
      </div>
    </main>
  );
}
