export default function SystemArchitecture() {
  const components = [
    { name: 'Camera', position: 'top-0 left-1/2 -translate-x-1/2' },
    { name: 'Vision Model', position: 'top-24 left-1/4' },
    { name: 'Control Logic', position: 'top-24 right-1/4' },
    { name: 'Robotic Arm', position: 'bottom-0 left-1/2 -translate-x-1/2' },
  ]

  return (
    <section className="py-20 bg-slate-50 dark:bg-slate-900">
      <div className="container mx-auto px-6">
        <h2 className="text-3xl md:text-4xl font-bold text-center text-slate-900 dark:text-white mb-4">
          System Architecture
        </h2>
        <p className="text-center text-slate-600 dark:text-slate-400 mb-16 max-w-2xl mx-auto">
          A modular pipeline connecting perception to action
        </p>
        <div className="max-w-4xl mx-auto">
          <div className="relative bg-white dark:bg-slate-800 rounded-xl p-12 shadow-lg">
            {/* Simple schematic diagram */}
            <div className="relative h-96">
              {components.map((component, index) => (
                <div
                  key={index}
                  className={`absolute ${component.position} bg-accent/10 border-2 border-accent rounded-lg px-6 py-4 text-center`}
                >
                  <div className="text-sm font-semibold text-accent">
                    {component.name}
                  </div>
                </div>
              ))}
              
              {/* Connection lines */}
              <svg className="absolute inset-0 w-full h-full pointer-events-none">
                <line
                  x1="50%"
                  y1="60"
                  x2="25%"
                  y2="120"
                  stroke="currentColor"
                  strokeWidth="2"
                  className="text-slate-300 dark:text-slate-600"
                />
                <line
                  x1="50%"
                  y1="60"
                  x2="75%"
                  y2="120"
                  stroke="currentColor"
                  strokeWidth="2"
                  className="text-slate-300 dark:text-slate-600"
                />
                <line
                  x1="25%"
                  y1="120"
                  x2="50%"
                  y2="320"
                  stroke="currentColor"
                  strokeWidth="2"
                  className="text-slate-300 dark:text-slate-600"
                />
                <line
                  x1="75%"
                  y1="120"
                  x2="50%"
                  y2="320"
                  stroke="currentColor"
                  strokeWidth="2"
                  className="text-slate-300 dark:text-slate-600"
                />
              </svg>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
