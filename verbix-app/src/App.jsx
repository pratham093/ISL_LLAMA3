import React, { useState, useEffect, useRef } from 'react'
import { Play, ChevronRight, Github, FileText, Moon, Sun, Eye, Mic, Globe, BookOpen, Users, Building, ArrowRight, Check, BarChart3, Database, Layers, Code, Hand, Brain, MessageSquare, Sparkles, ExternalLink, PlayCircle } from 'lucide-react'
import HelloImg from './Imgs/Hello.png'
import SorryImg from './Imgs/Sorry.png'
import FamilyImg from './Imgs/Family.png'
import PleaseImg from './Imgs/Please.png'
import MeImg from './Imgs/Me.png'
import ThanksImg from './Imgs/Thanks.png'
import DemoVideo from './Imgs/DRAFT(1).mp4'

const AnimatedCounter = ({ end, suffix = '' }) => {
  const [count, setCount] = useState(0)
  const ref = useRef(null)
  const [isVisible, setIsVisible] = useState(false)

  useEffect(() => {
    const observer = new IntersectionObserver(([entry]) => setIsVisible(entry.isIntersecting), { threshold: 0.1 })
    if (ref.current) observer.observe(ref.current)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    if (!isVisible) return
    let start = 0
    const increment = end / 125
    const timer = setInterval(() => {
      start += increment
      if (start >= end) { setCount(end); clearInterval(timer) }
      else { setCount(Math.floor(start)) }
    }, 16)
    return () => clearInterval(timer)
  }, [isVisible, end])

  return <span ref={ref}>{count}{suffix}</span>
}

const GestureCard = ({ gesture, word, meaning, image, darkMode }) => (
  <div className={`group relative overflow-hidden rounded-2xl p-8 min-h-[380px] transition-all duration-300 hover:scale-105 hover:shadow-2xl ${darkMode ? 'bg-gray-800/50 hover:bg-gray-800' : 'bg-white hover:bg-gray-50'} border ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
    <div className={`absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300 ${darkMode ? 'bg-gradient-to-br from-indigo-500/10 to-emerald-500/10' : 'bg-gradient-to-br from-indigo-100/50 to-emerald-100/50'}`} />
    <div className="relative z-10">
      <div className={`w-full h-60 mx-auto mb-6 rounded-2xl overflow-hidden ${darkMode ? 'bg-gray-900/60 border border-gray-700' : 'bg-gray-50 border border-gray-200'}`}>
        {image ? (
          <img src={image} alt={`${gesture} sign`} className="w-full h-full object-cover" />
        ) : (
          <div className="w-full h-full flex items-center justify-center">
            <Hand className={`w-12 h-12 ${darkMode ? 'text-gray-200' : 'text-gray-700'}`} />
          </div>
        )}
      </div>
      <div className="text-center">
        <div className={`text-xs font-medium uppercase tracking-wider mb-1 ${darkMode ? 'text-indigo-400' : 'text-indigo-600'}`}>Gesture</div>
        <div className={`text-base font-bold mb-2 ${darkMode ? 'text-white' : 'text-gray-900'}`}>{gesture}</div>
        <ArrowRight className={`w-4 h-4 mx-auto mb-2 ${darkMode ? 'text-gray-500' : 'text-gray-400'}`} />
        <div className={`text-xs font-medium uppercase tracking-wider mb-1 ${darkMode ? 'text-emerald-400' : 'text-emerald-600'}`}>Word</div>
        <div className={`text-base font-semibold mb-1 ${darkMode ? 'text-emerald-300' : 'text-emerald-700'}`}>{word}</div>
        <div className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>{meaning}</div>
      </div>
    </div>
  </div>
)

const StepCard = ({ icon: Icon, title, description, step, darkMode, color }) => (
  <div className={`relative p-8 rounded-2xl transition-all duration-300 hover:shadow-xl ${darkMode ? 'bg-gray-800/50 hover:bg-gray-800' : 'bg-white hover:shadow-2xl'} border ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
    <div className={`absolute -top-4 -left-4 w-10 h-10 rounded-full flex items-center justify-center text-white font-bold text-sm ${color}`}>{step}</div>
    <div className={`w-16 h-16 rounded-2xl flex items-center justify-center mb-6 ${color}`}>
      <Icon className="w-8 h-8 text-white" />
    </div>
    <h3 className={`text-xl font-bold mb-3 ${darkMode ? 'text-white' : 'text-gray-900'}`}>{title}</h3>
    <p className={`${darkMode ? 'text-gray-400' : 'text-gray-600'} leading-relaxed`}>{description}</p>
  </div>
)

export default function App() {
  const [darkMode, setDarkMode] = useState(true)

  const gestures = [
    { gesture: 'Wave Motion', word: 'Hello', meaning: 'A greeting gesture', image: HelloImg },
    { gesture: 'Hand on Heart', word: 'Sorry', meaning: 'Apology or regret', image: PleaseImg },
    { gesture: 'Circular Motion', word: 'Family', meaning: 'Group of relatives', image: FamilyImg },
    { gesture: 'Thumps Up(Right Hand) & Base (Left Hand)', word: 'Please', meaning: 'Polite request', image: SorryImg },
    { gesture: 'Pointing Self', word: 'I / Me', meaning: 'Self reference', image: MeImg },
    { gesture: 'Push the palm away from chin', word: 'Thanks', meaning: 'Expression of gratitude', image: ThanksImg },
  ]

  const stats = [
    { label: 'ISL Gestures', value: 30, suffix: '+' },
    { label: 'Accuracy', value: 94, suffix: '%' },
    { label: 'Training Sequences', value: 70, suffix: '' },
    { label: 'Frames per Sequence', value: 50, suffix: '' },
  ]

  const impacts = [
    { icon: Users, title: 'Accessibility', desc: 'Breaking communication barriers for 18M+ deaf individuals in India' },
    { icon: BookOpen, title: 'Education', desc: 'Learning tools for sign language students and educators' },
    { icon: Building, title: 'Public Services', desc: 'Enabling inclusive government and healthcare services' },
    { icon: Globe, title: 'Global Reach', desc: 'Connecting ISL users with the wider world' },
  ]

  const techStack = [
    { name: 'Python', icon: Code },
    { name: 'MediaPipe', icon: Hand },
    { name: 'TensorFlow', icon: Brain },
    { name: 'OpenCV', icon: Eye },
    { name: 'LLaMA-3', icon: MessageSquare },
    { name: 'Groq', icon: Sparkles },
  ]

  return (
    <div className={`min-h-screen transition-colors duration-300 ${darkMode ? 'bg-gray-950 text-white' : 'bg-gray-50 text-gray-900'}`}>
      <nav className={`fixed top-0 left-0 right-0 z-50 ${darkMode ? 'bg-gray-950/80' : 'bg-white/80'} backdrop-blur-xl border-b ${darkMode ? 'border-gray-800' : 'border-gray-200'}`}>
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <img src="/logo.png" alt="Verbix" className="w-10 h-10" />
              <span className="text-xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">Verbix</span>
            </div>
            <div className="hidden md:flex items-center gap-8">
              {['Problem', 'Solution', 'Demo', 'Technology'].map(item => (
                <a key={item} href={`#${item.toLowerCase()}`} className={`text-sm font-medium transition-colors ${darkMode ? 'text-gray-400 hover:text-white' : 'text-gray-600 hover:text-gray-900'}`}>{item}</a>
              ))}
            </div>
            <div className="flex items-center gap-4">
              <button onClick={() => setDarkMode(!darkMode)} className={`p-2 rounded-lg transition-colors ${darkMode ? 'bg-gray-800 hover:bg-gray-700' : 'bg-gray-100 hover:bg-gray-200'}`}>
                {darkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
              </button>
              <a href="https://github.com/pratham093/ISL_LLAMA3" target="_blank" rel="noopener noreferrer" className={`p-2 rounded-lg transition-colors ${darkMode ? 'bg-gray-800 hover:bg-gray-700' : 'bg-gray-100 hover:bg-gray-200'}`}>
                <Github className="w-5 h-5" />
              </a>
            </div>
          </div>
        </div>
      </nav>

      <section className="relative min-h-screen flex items-center justify-center overflow-hidden pt-20">
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className={`absolute top-1/4 left-1/4 w-96 h-96 rounded-full blur-3xl ${darkMode ? 'bg-indigo-500/10' : 'bg-indigo-500/5'}`} />
          <div className={`absolute bottom-1/4 right-1/4 w-96 h-96 rounded-full blur-3xl ${darkMode ? 'bg-violet-500/10' : 'bg-violet-500/5'}`} />
        </div>
        
        <div className="relative z-10 max-w-5xl mx-auto px-6 text-center">
          <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-full mb-8 ${darkMode ? 'bg-indigo-500/10 border border-indigo-500/20' : 'bg-indigo-100 border border-indigo-200'}`}>
            <Sparkles className={`w-4 h-4 ${darkMode ? 'text-indigo-400' : 'text-indigo-600'}`} />
            <span className={`text-sm font-medium ${darkMode ? 'text-indigo-300' : 'text-indigo-700'}`}>AI-Powered Sign Language Translation</span>
          </div>
          
          <h1 className="text-5xl md:text-7xl font-bold mb-6 leading-tight">
            <span className="bg-gradient-to-r from-indigo-400 via-violet-400 to-emerald-400 bg-clip-text text-transparent">Verbix</span>
          </h1>
          
          <p className={`text-xl md:text-2xl mb-4 font-light ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
            Translating Indian Sign Language into Meaningful Text using AI
          </p>
          
          <p className={`text-base md:text-lg mb-10 max-w-3xl mx-auto ${darkMode ? 'text-gray-500' : 'text-gray-600'}`}>
            Real-time gesture recognition powered by MediaPipe, LSTM sequence learning, and LLaMA-3 language intelligence.
          </p>
          
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <a href="#demo" className="group px-8 py-4 rounded-xl bg-gradient-to-r from-indigo-500 to-violet-600 text-white font-semibold transition-all duration-300 hover:shadow-lg hover:shadow-indigo-500/30 hover:scale-105 flex items-center gap-2">
              <Play className="w-5 h-5" />
              Watch Demo
              <ChevronRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </a>
            <a href="https://github.com/pratham093/ISL_LLAMA3" target="_blank" rel="noopener noreferrer" className={`px-8 py-4 rounded-xl font-semibold transition-all duration-300 flex items-center gap-2 ${darkMode ? 'bg-gray-800 hover:bg-gray-700 text-white' : 'bg-white hover:bg-gray-100 text-gray-900 border border-gray-200'}`}>
              <Github className="w-5 h-5" />
              View Source
            </a>
          </div>

          <div className="mt-16 grid grid-cols-2 md:grid-cols-4 gap-8 md:gap-16">
            {stats.map((stat, i) => (
              <div key={i} className="text-center">
                <div className={`text-3xl md:text-4xl font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                  <AnimatedCounter end={stat.value} suffix={stat.suffix} />
                </div>
                <div className={`text-sm ${darkMode ? 'text-gray-500' : 'text-gray-600'}`}>{stat.label}</div>
              </div>
            ))}
          </div>
        </div>

        <div className={`absolute bottom-10 left-1/2 -translate-x-1/2 animate-bounce ${darkMode ? 'text-gray-600' : 'text-gray-400'}`}>
          <ChevronRight className="w-6 h-6 rotate-90" />
        </div>
      </section>

      <section id="problem" className={`py-24 ${darkMode ? 'bg-gray-900/50' : 'bg-white'}`}>
        <div className="max-w-7xl mx-auto px-6">
          <div className="grid lg:grid-cols-2 gap-16 items-center">
            <div>
              <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-sm font-medium mb-6 ${darkMode ? 'bg-red-500/10 text-red-400' : 'bg-red-100 text-red-700'}`}>The Problem</div>
              <h2 className={`text-4xl font-bold mb-6 ${darkMode ? 'text-white' : 'text-gray-900'}`}>Breaking Communication Barriers</h2>
              <p className={`text-lg mb-6 leading-relaxed ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                Over 18 million people in India are deaf or hard of hearing, yet most digital communication systems remain inaccessible. The lack of real-time, contextual Indian Sign Language translation tools creates daily barriers in education, healthcare, and public services.
              </p>
              <p className={`text-lg mb-8 leading-relaxed ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                Existing solutions often fail to capture the dynamic, temporal nature of sign language or produce grammatically correct sentences. Verbix bridges this gap with cutting-edge AI.
              </p>
              <div className="flex flex-wrap gap-4">
                {['Real-time Translation', 'Contextual Understanding', 'Natural Language Output'].map((item, i) => (
                  <div key={i} className={`flex items-center gap-2 px-4 py-2 rounded-lg ${darkMode ? 'bg-gray-800' : 'bg-gray-100'}`}>
                    <Check className={`w-4 h-4 ${darkMode ? 'text-emerald-400' : 'text-emerald-600'}`} />
                    <span className={`text-sm font-medium ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>{item}</span>
                  </div>
                ))}
              </div>
            </div>
            <div className="relative">
              <div className={`aspect-square rounded-3xl overflow-hidden ${darkMode ? 'bg-gradient-to-br from-indigo-900/50 to-violet-900/50' : 'bg-gradient-to-br from-indigo-100 to-violet-100'} p-8`}>
                <div className="h-full flex flex-col justify-center items-center gap-6">
                  <div className={`w-24 h-24 rounded-2xl flex items-center justify-center ${darkMode ? 'bg-indigo-600' : 'bg-indigo-500'} animate-float`}>
                    <Hand className="w-12 h-12 text-white" />
                  </div>
                  <ArrowRight className={`w-8 h-8 rotate-90 ${darkMode ? 'text-gray-600' : 'text-gray-400'}`} />
                  <div className={`w-24 h-24 rounded-2xl flex items-center justify-center ${darkMode ? 'bg-violet-600' : 'bg-violet-500'} animate-float`} style={{ animationDelay: '0.5s' }}>
                    <Brain className="w-12 h-12 text-white" />
                  </div>
                  <ArrowRight className={`w-8 h-8 rotate-90 ${darkMode ? 'text-gray-600' : 'text-gray-400'}`} />
                  <div className={`w-24 h-24 rounded-2xl flex items-center justify-center ${darkMode ? 'bg-emerald-600' : 'bg-emerald-500'} animate-float`} style={{ animationDelay: '1s' }}>
                    <MessageSquare className="w-12 h-12 text-white" />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section id="solution" className="py-24">
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center mb-16">
            <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-sm font-medium mb-6 ${darkMode ? 'bg-indigo-500/10 text-indigo-400' : 'bg-indigo-100 text-indigo-700'}`}>The Solution</div>
            <h2 className={`text-4xl font-bold mb-4 ${darkMode ? 'text-white' : 'text-gray-900'}`}>How Verbix Works</h2>
            <p className={`text-lg max-w-2xl mx-auto ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
              A three-stage AI pipeline that transforms hand gestures into meaningful natural language
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            <StepCard
              icon={Eye}
              title="Gesture Capture"
              description="MediaPipe Holistic extracts 1,662 keypoints per frame from hands, pose, and face landmarks, creating a detailed spatial representation of each gesture."
              step="1"
              darkMode={darkMode}
              color="bg-gradient-to-br from-blue-500 to-cyan-500"
            />
            <StepCard
              icon={Brain}
              title="Temporal Learning"
              description="LSTM neural networks process sequences of 50 frames, learning the temporal patterns that distinguish different ISL gestures and converting them to words."
              step="2"
              darkMode={darkMode}
              color="bg-gradient-to-br from-violet-500 to-purple-500"
            />
            <StepCard
              icon={MessageSquare}
              title="Language Intelligence"
              description="LLaMA-3 (via Groq) transforms recognized word sequences into grammatically correct, contextually aware sentences that convey the intended meaning."
              step="3"
              darkMode={darkMode}
              color="bg-gradient-to-br from-emerald-500 to-teal-500"
            />
          </div>
        </div>
      </section>

      <section id="demo" className={`py-24 ${darkMode ? 'bg-gray-900/50' : 'bg-white'}`}>
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center mb-16">
            <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-sm font-medium mb-6 ${darkMode ? 'bg-violet-500/10 text-violet-400' : 'bg-violet-100 text-violet-700'}`}>See It In Action</div>
            <h2 className={`text-4xl font-bold mb-4 ${darkMode ? 'text-white' : 'text-gray-900'}`}>Live Demo</h2>
            <p className={`text-lg max-w-2xl mx-auto ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
              Watch Verbix translate Indian Sign Language gestures into natural sentences in real-time
            </p>
          </div>

          <div className={`max-w-4xl mx-auto rounded-3xl overflow-hidden ${darkMode ? 'bg-gray-800' : 'bg-white shadow-2xl'} border ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
            <div className="aspect-video relative bg-black">
              <video
                className="absolute inset-0 w-full h-full object-cover"
                src={DemoVideo}
                controls
                preload="metadata"
              />
              
              {/* 
              YOUTUBE EMBED - Uncomment and replace VIDEO_ID with your YouTube video ID:
              
              <iframe 
                className="absolute inset-0 w-full h-full"
                src="https://www.youtube.com/embed/VIDEO_ID"
                title="Verbix Demo"
                frameBorder="0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
              />
              */}
            </div>
            
            <div className={`p-6 ${darkMode ? 'bg-gray-800' : 'bg-gray-50'}`}>
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div>
                  <h3 className={`font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>Try It Yourself</h3>
                  <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>Clone the repository and run locally</p>
                </div>
                <a href="https://github.com/pratham093/ISL_LLAMA3" target="_blank" rel="noopener noreferrer" className="px-6 py-3 rounded-xl bg-gradient-to-r from-indigo-500 to-violet-600 text-white font-semibold flex items-center gap-2 hover:shadow-lg transition-all">
                  <Github className="w-5 h-5" />
                  View on GitHub
                  <ExternalLink className="w-4 h-4" />
                </a>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="py-24">
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center mb-16">
            <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-sm font-medium mb-6 ${darkMode ? 'bg-emerald-500/10 text-emerald-400' : 'bg-emerald-100 text-emerald-700'}`}>Gesture Library</div>
            <h2 className={`text-4xl font-bold mb-4 ${darkMode ? 'text-white' : 'text-gray-900'}`}>Supported Gestures</h2>
            <p className={`text-lg max-w-2xl mx-auto ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
              Verbix recognizes 30+ common Indian Sign Language gestures
            </p>
          </div>

          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {gestures.map((g, i) => (
              <GestureCard key={i} {...g} darkMode={darkMode} />
            ))}
          </div>
        </div>
      </section>

      <section id="technology" className={`py-24 ${darkMode ? 'bg-gray-900/50' : 'bg-white'}`}>
        <div className="max-w-7xl mx-auto px-6">
          <div className="grid lg:grid-cols-2 gap-16">
            <div>
              <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-sm font-medium mb-6 ${darkMode ? 'bg-blue-500/10 text-blue-400' : 'bg-blue-100 text-blue-700'}`}>Training</div>
              <h2 className={`text-4xl font-bold mb-8 ${darkMode ? 'text-white' : 'text-gray-900'}`}>Dataset & Model</h2>
              
              <div className={`p-6 rounded-2xl mb-6 ${darkMode ? 'bg-gray-800' : 'bg-gray-50'} border ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                <div className="flex items-center gap-4 mb-4">
                  <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center">
                    <Database className="w-6 h-6 text-white" />
                  </div>
                  <h3 className={`text-lg font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>Dataset Specs</h3>
                </div>
                <div className="space-y-3">
                  {[
                    { label: 'Total Gestures', value: '30 ISL Words' },
                    { label: 'Sequences per Gesture', value: '70 Sequences' },
                    { label: 'Frames per Sequence', value: '50 Frames' },
                    { label: 'Keypoints per Frame', value: '1,662 Points' },
                  ].map((item, i) => (
                    <div key={i} className={`flex justify-between py-2 border-b ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                      <span className={darkMode ? 'text-gray-400' : 'text-gray-600'}>{item.label}</span>
                      <span className={`font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>{item.value}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className={`p-6 rounded-2xl ${darkMode ? 'bg-gray-800' : 'bg-gray-50'} border ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                <div className="flex items-center gap-4 mb-4">
                  <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-500 flex items-center justify-center">
                    <BarChart3 className="w-6 h-6 text-white" />
                  </div>
                  <h3 className={`text-lg font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>Model Performance</h3>
                </div>
                <div className="space-y-4">
                  {[
                    { label: 'Training Accuracy', value: 98, color: 'from-emerald-500 to-teal-500' },
                    { label: 'Validation Accuracy', value: 94, color: 'from-blue-500 to-cyan-500' },
                    { label: 'F1 Score', value: 93, color: 'from-violet-500 to-purple-500' },
                  ].map((item, i) => (
                    <div key={i}>
                      <div className="flex justify-between mb-2">
                        <span className={darkMode ? 'text-gray-400' : 'text-gray-600'}>{item.label}</span>
                        <span className={`font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>{item.value}%</span>
                      </div>
                      <div className={`h-2 rounded-full ${darkMode ? 'bg-gray-700' : 'bg-gray-200'} overflow-hidden`}>
                        <div className={`h-full rounded-full bg-gradient-to-r ${item.color}`} style={{ width: `${item.value}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div>
              <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-sm font-medium mb-6 ${darkMode ? 'bg-purple-500/10 text-purple-400' : 'bg-purple-100 text-purple-700'}`}>Architecture</div>
              <h2 className={`text-4xl font-bold mb-8 ${darkMode ? 'text-white' : 'text-gray-900'}`}>System Pipeline</h2>
              
              <div className={`p-8 rounded-2xl ${darkMode ? 'bg-gray-800' : 'bg-gray-50'} border ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                <div className="space-y-6">
                  {[
                    { icon: Eye, label: 'Camera Input', desc: 'Real-time video capture at 30 FPS', color: 'from-gray-500 to-gray-600' },
                    { icon: Hand, label: 'MediaPipe Holistic', desc: 'Extract 1,662 keypoints per frame', color: 'from-blue-500 to-cyan-500' },
                    { icon: Layers, label: 'Sequence Buffer', desc: 'Collect 50 frames for temporal analysis', color: 'from-indigo-500 to-blue-500' },
                    { icon: Brain, label: 'LSTM Network', desc: 'Classify gesture sequences into words', color: 'from-violet-500 to-purple-500' },
                    { icon: Sparkles, label: 'LLaMA-3 (Groq)', desc: 'Generate natural language sentences', color: 'from-emerald-500 to-teal-500' },
                    { icon: MessageSquare, label: 'Text Output', desc: 'Display translated sentence', color: 'from-pink-500 to-rose-500' },
                  ].map((item, i, arr) => (
                    <div key={i} className="flex items-start gap-4">
                      <div className={`w-12 h-12 rounded-xl flex items-center justify-center bg-gradient-to-br ${item.color} flex-shrink-0`}>
                        <item.icon className="w-6 h-6 text-white" />
                      </div>
                      <div className="flex-1 pt-1">
                        <h4 className={`font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>{item.label}</h4>
                        <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>{item.desc}</p>
                        {i < arr.length - 1 && (
                          <div className={`w-0.5 h-6 ml-6 mt-4 ${darkMode ? 'bg-gray-700' : 'bg-gray-300'}`} />
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="py-24">
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center mb-16">
            <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-sm font-medium mb-6 ${darkMode ? 'bg-teal-500/10 text-teal-400' : 'bg-teal-100 text-teal-700'}`}>Impact</div>
            <h2 className={`text-4xl font-bold mb-4 ${darkMode ? 'text-white' : 'text-gray-900'}`}>Real-World Applications</h2>
          </div>

          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {impacts.map((impact, i) => (
              <div key={i} className={`p-6 rounded-2xl transition-all duration-300 hover:scale-105 ${darkMode ? 'bg-gray-800/50 hover:bg-gray-800' : 'bg-white hover:shadow-xl'} border ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                <div className="w-14 h-14 rounded-xl flex items-center justify-center mb-4 bg-gradient-to-br from-teal-500 to-emerald-500">
                  <impact.icon className="w-7 h-7 text-white" />
                </div>
                <h3 className={`text-lg font-bold mb-2 ${darkMode ? 'text-white' : 'text-gray-900'}`}>{impact.title}</h3>
                <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>{impact.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className={`py-24 ${darkMode ? 'bg-gray-900/50' : 'bg-white'}`}>
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center mb-16">
            <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-sm font-medium mb-6 ${darkMode ? 'bg-orange-500/10 text-orange-400' : 'bg-orange-100 text-orange-700'}`}>Built With</div>
            <h2 className={`text-4xl font-bold mb-4 ${darkMode ? 'text-white' : 'text-gray-900'}`}>Technology Stack</h2>
          </div>

          <div className="flex flex-wrap justify-center gap-4">
            {techStack.map((tech, i) => (
              <div key={i} className={`flex items-center gap-3 px-6 py-4 rounded-xl transition-all duration-300 hover:scale-105 ${darkMode ? 'bg-gray-800 hover:bg-gray-750' : 'bg-gray-50 hover:shadow-lg'} border ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                <tech.icon className={`w-6 h-6 ${darkMode ? 'text-indigo-400' : 'text-indigo-600'}`} />
                <span className={`font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>{tech.name}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="py-24">
        <div className="max-w-4xl mx-auto px-6 text-center">
          <div className={`p-12 rounded-3xl ${darkMode ? 'bg-gradient-to-br from-indigo-900/50 to-violet-900/50 border border-indigo-800' : 'bg-gradient-to-br from-indigo-100 to-violet-100 border border-indigo-200'}`}>
            <h2 className={`text-3xl md:text-4xl font-bold mb-4 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
              Ready to Explore?
            </h2>
            <p className={`text-lg mb-8 ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
              Clone the repository, set up your environment, and start translating sign language.
            </p>
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <a href="https://github.com/pratham093/ISL_LLAMA3" target="_blank" rel="noopener noreferrer" className="px-8 py-4 rounded-xl bg-gradient-to-r from-indigo-500 to-violet-600 text-white font-semibold transition-all duration-300 hover:shadow-lg hover:shadow-indigo-500/30 hover:scale-105 flex items-center gap-2">
                <Github className="w-5 h-5" />
                View on GitHub
              </a>
              <a href="#demo" className={`px-8 py-4 rounded-xl font-semibold transition-all duration-300 flex items-center gap-2 ${darkMode ? 'bg-white/10 hover:bg-white/20 text-white' : 'bg-white hover:bg-gray-50 text-gray-900 shadow-lg'}`}>
                <Play className="w-5 h-5" />
                Watch Demo
              </a>
            </div>
          </div>
        </div>
      </section>

      <footer className={`py-12 border-t ${darkMode ? 'border-gray-800' : 'border-gray-200'}`}>
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex flex-col md:flex-row items-center justify-between gap-6">
            <div className="flex items-center gap-3">
              <img src="/logo.png" alt="Verbix" className="w-10 h-10" />
              <span className="text-xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">Verbix</span>
            </div>
            <div className="flex items-center gap-6">
              <a href="https://github.com/pratham093/ISL_LLAMA3" target="_blank" rel="noopener noreferrer" className={`flex items-center gap-2 transition-colors ${darkMode ? 'text-gray-400 hover:text-white' : 'text-gray-600 hover:text-gray-900'}`}>
                <Github className="w-5 h-5" />
                <span className="text-sm">GitHub</span>
              </a>
            </div>
            <p className={`text-sm ${darkMode ? 'text-gray-500' : 'text-gray-600'}`}>
              © 2024 Verbix. Built for accessibility.
            </p>
          </div>
        </div>
      </footer>
    </div>
  )
}
