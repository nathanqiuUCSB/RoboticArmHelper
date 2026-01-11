export default function Hero() {
  return (
    <section className="relative min-h-screen flex items-center justify-center bg-gradient-to-b from-slate-50 to-white dark:from-slate-900 dark:to-slate-800">
      <div className="container mx-auto px-6 py-20 text-center">
        <h1 className="text-5xl md:text-7xl font-bold text-slate-900 dark:text-white mb-6">
          Vision-Guided
          <br />
          <span className="text-accent">Robotic Arm</span>
        </h1>
        <p className="text-xl md:text-2xl text-slate-600 dark:text-slate-300 mb-10 max-w-3xl mx-auto">
          A vision-guided robotic arm that detects and grasps real-world objects in real time.
        </p>
        <a
          href="#demo"
          className="inline-block px-8 py-4 bg-accent hover:bg-accent-dark text-white font-semibold rounded-lg shadow-lg transition-all duration-200 hover:shadow-xl hover:scale-105"
        >
          Watch Demo
        </a>
      </div>
    </section>
  )
}
