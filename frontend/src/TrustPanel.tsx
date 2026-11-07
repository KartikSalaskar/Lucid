import { useEffect, useState } from 'react'

type TrustResponse = {
  verdict: string
  label: string
  reasons: string[]
  signals: {
    quality_score: number | null
    calibrated_confidence: number | null
    uncertainty: number | null
    ood_score: number | null
    ood_flag: boolean | null
  }
  artifacts: Record<string, boolean>
  disclaimer: string
}

export default function TrustPanel({ file }: { file: File | null }) {
  const [data, setData] = useState<TrustResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    const controller = new AbortController()

    setData(null)
    setError('')

    if (!file) {
      setLoading(false)
      return () => controller.abort()
    }

    async function assess() {
      setLoading(true)

      try {
        const form = new FormData()
        form.append('file', file!)

        const response = await fetch('/api/trust/assess', {
          method: 'POST',
          body: form,
          signal: controller.signal
        })

        const result = await response.json()

        if (!response.ok) {
          throw new Error(result.detail || 'Trust assessment failed')
        }

        setData(result as TrustResponse)
      } catch (e) {
        if (!controller.signal.aborted) {
          setError((e as Error).message)
        }
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false)
        }
      }
    }

    assess()

    return () => controller.abort()
  }, [file])

  return (
    <div className="card">
      <h2>Trust Signals &amp; Reliability</h2>

      {!file && <p>Upload an image to assess its reliability.</p>}
      {loading && <p>Checking trust signals...</p>}
      {error && <p className="err">{error}</p>}

      {data && (
        <>
          <h3>{data.label}</h3>

          <div className="row">
            <span>Image quality</span>
            <span>
              {data.signals.quality_score === null
                ? 'Unavailable'
                : `${data.signals.quality_score}/100`}
            </span>
          </div>

          <div className="row">
            <span>Calibrated confidence</span>
            <span>
              {data.signals.calibrated_confidence === null
                ? 'Unavailable'
                : `${(data.signals.calibrated_confidence * 100).toFixed(1)}%`}
            </span>
          </div>

          <div className="row">
            <span>Ensemble uncertainty (std)</span>
            <span>
              {data.signals.uncertainty === null
                ? 'Unavailable'
                : data.signals.uncertainty.toFixed(4)}
            </span>
          </div>

          <div className="row">
            <span>OOD score</span>
            <span>
              {data.signals.ood_score === null
                ? 'Unavailable'
                : data.signals.ood_score.toFixed(4)}
            </span>
          </div>

          <div className="row">
            <span>OOD flag</span>
            <span>
              {data.signals.ood_flag === null
                ? 'Unavailable'
                : data.signals.ood_flag ? 'Flagged' : 'Not flagged'}
            </span>
          </div>

          <h3>Artifact availability</h3>
          {Object.entries(data.artifacts).map(([name, available]) => (
            <div className="row" key={name}>
              <span>{name}</span>
              <span>{available ? 'Available' : 'Unavailable'}</span>
            </div>
          ))}

          <h3>Reasons</h3>
          <ul>
            {data.reasons.map((reason, index) => (
              <li key={index}>{reason}</li>
            ))}
          </ul>

          <p>{data.disclaimer}</p>
        </>
      )}
    </div>
  )
}
