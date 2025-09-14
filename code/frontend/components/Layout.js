import Link from 'next/link'
import { useRouter } from 'next/router'
import { Search, BarChart3, Database, Settings, FileText } from 'lucide-react'

export default function Layout({ children }) {
  const router = useRouter()
  
  const navigation = [
    { name: 'Search', href: '/', icon: Search },
    { name: 'Analysis', href: '/analysis', icon: BarChart3 },
    { name: 'Sample Queries', href: '/sample-queries', icon: FileText },
    { name: 'Database', href: '/database', icon: Database },
    { name: 'Config', href: '/config', icon: Settings },
  ]
  
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Navigation */}
      <nav className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex">
              <div className="flex-shrink-0 flex items-center">
                <h1 className="text-xl font-bold text-gray-900">
                  Vietnamese-to-SQL Thesis
                </h1>
              </div>
              <div className="hidden sm:ml-6 sm:flex sm:space-x-8">
                {navigation.map((item) => {
                  const Icon = item.icon
                  const isActive = router.pathname === item.href
                  return (
                    <Link
                      key={item.name}
                      href={item.href}
                      className={`inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium ${
                        isActive
                          ? 'border-primary-500 text-primary-600'
                          : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                      }`}
                    >
                      <Icon className="w-4 h-4 mr-2" />
                      {item.name}
                    </Link>
                  )
                })}
              </div>
            </div>
            
          </div>
        </div>
      </nav>
      
      {/* Main content */}
      <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        {children}
      </main>
    </div>
  )
}
