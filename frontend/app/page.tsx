import Hero from '@/components/Hero'
import VideoDemo from '@/components/VideoDemo'
import HowItWorks from '@/components/HowItWorks'
import KeyFeatures from '@/components/KeyFeatures'
import SystemArchitecture from '@/components/SystemArchitecture'
import AboutTeam from '@/components/AboutTeam'
import TechStack from '@/components/TechStack'
import Footer from '@/components/Footer'

export default function Home() {
  return (
    <main className="min-h-screen">
      <Hero />
      <VideoDemo />
      <HowItWorks />
      <KeyFeatures />
      <SystemArchitecture />
      <AboutTeam />
      <TechStack />
      <Footer />
    </main>
  )
}
