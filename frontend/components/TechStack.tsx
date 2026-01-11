export default function TechStack() {
  const technologies = [
    { name: 'YOLOv8', category: 'Vision Model' },
    { name: 'OpenCV', category: 'Computer Vision' },
    { name: 'Python', category: 'Backend' },
    { name: 'React', category: 'Frontend' },
    { name: 'Next.js', category: 'Frontend' },
    { name: 'LeRobot', category: 'Robotics' },
    { name: 'Groq/Gemini', category: 'AI Planning' },
  ]

  return (
    <section className="py-20 bg-slate-50 dark:bg-slate-900">
      <div className="container mx-auto px-6">
        <h2 className="text-3xl md:text-4xl font-bold text-center text-slate-900 dark:text-white mb-4">
          Tech Stack
        </h2>
        <p className="text-center text-slate-600 dark:text-slate-400 mb-16 max-w-2xl mx-auto">
          Built with modern, open-source technologies
        </p>
        <div className="flex flex-wrap justify-center gap-4 max-w-4xl mx-auto">
          {technologies.map((tech, index) => (
            <div
              key={index}
              className="bg-white dark:bg-slate-800 border-2 border-slate-200 dark:border-slate-700 rounded-full px-6 py-3 hover:border-accent hover:shadow-md transition-all duration-200"
            >
              <span className="text-slate-900 dark:text-white font-medium">
                {tech.name}
              </span>
              <span className="text-slate-500 dark:text-slate-400 text-sm ml-2">
                {tech.category}
              </span>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
