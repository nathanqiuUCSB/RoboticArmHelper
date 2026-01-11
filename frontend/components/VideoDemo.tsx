export default function VideoDemo() {
  return (
    <section id="demo" className="py-20 bg-white dark:bg-slate-800">
      <div className="container mx-auto px-6">
        <h2 className="text-3xl md:text-4xl font-bold text-center text-slate-900 dark:text-white mb-4">
          Live Demo
        </h2>
        <p className="text-center text-slate-600 dark:text-slate-400 mb-12">
          Live object detection and grasping demo
        </p>
        <div className="max-w-5xl mx-auto">
          <div className="relative aspect-video bg-slate-200 dark:bg-slate-700 rounded-xl overflow-hidden shadow-2xl">
            {/* Placeholder for video - replace with actual YouTube embed or video */}
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-center">
                <svg
                  className="w-24 h-24 mx-auto text-slate-400 dark:text-slate-500 mb-4"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"
                  />
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
                <p className="text-slate-500 dark:text-slate-400">
                  Video placeholder - embed YouTube or local MP4 here
                </p>
                {/* Uncomment and replace with actual video:
                <iframe
                  className="absolute inset-0 w-full h-full"
                  src="https://www.youtube.com/embed/YOUR_VIDEO_ID"
                  title="Robotic Arm Demo"
                  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                  allowFullScreen
                ></iframe>
                */}
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
