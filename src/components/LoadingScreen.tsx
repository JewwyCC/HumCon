import { useEffect, useState } from "react";

const LoadingScreen = () => {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          clearInterval(timer);
          return 100;
        }
        return prev + 2;
      });
    }, 50);

    return () => clearInterval(timer);
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary/10 via-secondary/10 to-accent/10 flex items-center justify-center">
      <div className="text-center space-y-8">
        {/* Logo */}
        <div className="space-y-4">
          <h1 className="text-6xl font-bold gradient-text">HumCon</h1>
          <p className="text-xl text-muted-foreground">
            Human Content Revolution
          </p>
        </div>

        {/* Loading Animation */}
        <div className="space-y-4">
          <div className="w-64 h-2 bg-muted rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-primary to-secondary transition-all duration-100 ease-out"
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="text-sm text-muted-foreground">
            Loading the future of authentic content...
          </p>
        </div>

        {/* Floating Icons */}
        <div className="relative">
          <div className="absolute -top-20 -left-20 w-8 h-8 bg-primary/20 rounded-full float" />
          <div className="absolute -top-16 -right-16 w-6 h-6 bg-secondary/20 rounded-full float" style={{ animationDelay: '1s' }} />
          <div className="absolute -bottom-16 -left-16 w-10 h-10 bg-accent/20 rounded-full float" style={{ animationDelay: '2s' }} />
        </div>
      </div>
    </div>
  );
};

export default LoadingScreen;