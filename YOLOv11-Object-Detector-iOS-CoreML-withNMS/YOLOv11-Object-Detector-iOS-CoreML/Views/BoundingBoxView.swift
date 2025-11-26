import SwiftUI

struct BoundingBoxView: View {
    let object: BoundingBox
    /// Kích thước phần ảnh HIỂN THỊ (không phải toàn màn)
    let parentSize: CGSize
    /// Offset của ảnh trong khung màn hình (dùng cho ảnh upload – letterbox)
    let offset: CGPoint
    
    init(object: BoundingBox, parentSize: CGSize, offset: CGPoint = .zero) {
        self.object = object
        self.parentSize = parentSize
        self.offset = offset
    }
    
    // nếu muốn set màu riêng từng lớp thì bỏ vô đây
    static let classColors: [String: Color] = [:]
    
    private func color(for label: String) -> Color {
        if let c = Self.classColors[label] {
            return c
        }
        // random stable color theo tên label
        var hasher = Hasher()
        hasher.combine(label)
        let hash = hasher.finalize()
        let r = Double((hash & 0xFF0000) >> 16) / 255.0
        let g = Double((hash & 0x00FF00) >> 8) / 255.0
        let b = Double(hash & 0x0000FF) / 255.0
        return Color(red: r, green: g, blue: b)
    }
    
    private var rect: CGRect {
        // YOLO trả boundingBox (x, y, w, h) trong hệ VN:
        // x,y = bottom-left, normalized 0–1, y đếm từ BÊN DƯỚI lên
        let bbox = object.boundingBox
        
        let w = bbox.width  * parentSize.width
        let h = bbox.height * parentSize.height
        
        let x = bbox.minX * parentSize.width
        // VN to SwiftUI: origin (0,0) ở trên trái -> phải lật trục Y
        let y = (1 - bbox.maxY) * parentSize.height
        
        return CGRect(x: offset.x + x,
                      y: offset.y + y,
                      width:  w,
                      height: h)
    }
    
    var body: some View {
        let boxColor = color(for: object.label)
        let rect = self.rect
        
        ZStack(alignment: .topLeading) {
            Rectangle()
                .stroke(boxColor, lineWidth: 2)
                .frame(width: rect.width, height: rect.height)
                .position(x: rect.midX, y: rect.midY)
            
            Text(object.label + ": " + String(format: "%.0f%%", object.confidence * 100))
                .font(.caption2)
                .foregroundColor(.white)
                .padding(5)
                .background(boxColor.opacity(0.9))
                .cornerRadius(8)
                .position(x: rect.minX + 4 + rect.width / 2,
                          y: max(rect.minY - 12, 0))
        }
    }
}

