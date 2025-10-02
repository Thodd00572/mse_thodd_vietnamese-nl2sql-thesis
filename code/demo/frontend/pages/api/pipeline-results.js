// API endpoint to serve pipeline results
export default function handler(req, res) {
  // Mock data based on the actual results you provided
  const mockData = {
    'P1': {
      pipeline: 'P1_Prompting_mT5',
      name: 'P1: mT5 Zero-shot',
      N: 300,
      EM: 0.16,
      EX: 0.327,
      ErrorRate: 0.013,
      Latency_mean: 0.299,
      GPU_peak_GB: 2.47,
      Model_Success_Rate: 1.0,
      Success_Rate_Simple: 1.0,
      Success_Rate_Medium: 1.0,
      Success_Rate_Complex: 1.0
    },
    'P2': {
      pipeline: 'SQLCoder',
      name: 'P2: SQLCoder Zero-shot',
      N: 300,
      EM: 0.18,
      EX: 0.223,
      ErrorRate: 0.0,
      Latency_mean: 1.76,
      GPU_peak_GB: 13.82,
      Model_Success_Rate: 1.0,
      Success_Rate_Simple: 1.0,
      Success_Rate_Medium: 1.0,
      Success_Rate_Complex: 1.0
    },
    'P3': {
      pipeline: 'P5_Vanna_AI',
      name: 'P3: Vanna AI RAG',
      N: 100,
      EM: 0.77,
      EX: 0.82,
      ErrorRate: 0.1,
      Latency_mean: 2.31,
      GPU_peak_GB: 0.0,
      Model_Success_Rate: 0.9,
      Success_Rate_Simple: 0.9,
      Success_Rate_Medium: 0.0,
      Success_Rate_Complex: 0.0
    }
  }

  if (req.method === 'GET') {
    res.status(200).json(mockData)
  } else {
    res.setHeader('Allow', ['GET'])
    res.status(405).end(`Method ${req.method} Not Allowed`)
  }
}
