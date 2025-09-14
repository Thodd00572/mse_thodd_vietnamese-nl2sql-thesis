import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line } from 'recharts'

export function ComparisonChart({ data, title }) {
  return (
    <div className="w-full h-80">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">{title}</h3>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Bar dataKey="Pipeline1" fill="#3b82f6" name="Pipeline 1" />
          <Bar dataKey="Pipeline2" fill="#22c55e" name="Pipeline 2" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

export function PerformanceChart({ data, title }) {
  return (
    <div className="w-full h-80">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">{title}</h3>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="Pipeline1" stroke="#3b82f6" strokeWidth={3} />
          <Line type="monotone" dataKey="Pipeline2" stroke="#22c55e" strokeWidth={3} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
