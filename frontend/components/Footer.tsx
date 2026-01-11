export default function Footer() {
  return (
    <footer className="bg-slate-900 dark:bg-black text-white py-12">
      <div className="container mx-auto px-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-8">
          <div>
            <h3 className="text-2xl font-bold mb-4">Vision-Guided Robotic Arm</h3>
            <p className="text-slate-400">
              Autonomous object detection and grasping through computer vision
            </p>
          </div>
          <div>
            <h4 className="font-semibold mb-4">Project</h4>
            <ul className="space-y-2 text-slate-400">
              <li>
                <a href="#demo" className="hover:text-white transition-colors">
                  Demo
                </a>
              </li>
              <li>
                <a href="#features" className="hover:text-white transition-colors">
                  Features
                </a>
              </li>
              <li>
                <a href="#team" className="hover:text-white transition-colors">
                  Team
                </a>
              </li>
            </ul>
          </div>
          <div>
            <h4 className="font-semibold mb-4">Links</h4>
            <ul className="space-y-2 text-slate-400">
              <li>
                <a
                  href="https://github.com/yourusername/robotic-arm-helper"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-white transition-colors"
                >
                  GitHub Repository
                </a>
              </li>
              <li className="text-slate-500">Hackathon 2024</li>
            </ul>
          </div>
        </div>
        <div className="border-t border-slate-800 pt-8 text-center text-slate-500">
          <p>&copy; {new Date().getFullYear()} Vision-Guided Robotic Arm. Built for hackathon demo.</p>
        </div>
      </div>
    </footer>
  )
}
