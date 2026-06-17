import { useState } from 'react'

// MJPEG 라이브 피드. <img> 가 multipart/x-mixed-replace 를 네이티브로 디코드.
// 영상은 WS(JSON)와 완전 분리된 HTTP 채널이라 그래프 갱신과 간섭 없음.
export function CameraFeed({ available }: { available: boolean }) {
  // available 토글 시 src 를 강제 리프레시 (캐시 끊고 재연결)
  const [nonce] = useState(() => Date.now())
  if (!available) {
    return (
      <div className="camera">
        <div className="placeholder">카메라 스트림 대기 중…<br />(vision 노드 미연결)</div>
      </div>
    )
  }
  return (
    <div className="camera">
      <img src={`/api/camera/stream?t=${nonce}`} alt="camera live feed" />
    </div>
  )
}
