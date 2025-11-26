import Foundation
import Vision
import CoreML
import CoreVideo

@Observable @MainActor
final class MainDishClassifier {
    private let visionModel: VNCoreMLModel?
    
    /// Kết quả phân loại món chính
    var mainDishLabel: String?
    var mainDishConfidence: Float = 0.0
    
    /// Throttle cho realtime camera
    private var lastUpdateTime: CFAbsoluteTime = 0
    private let updateInterval: CFAbsoluteTime = 0.7
    
    /// Ngưỡng tin cậy tối thiểu để show Món chính
    private let minDishConfidence: Float = 0.4
    
    init(modelName: String) {
        do {
            let config = MLModelConfiguration()
            // ĐỔI TÊN CLASS NÀY CHO ĐÚNG VỚI .mlmodel (ví dụ FoodMainDishEfficientNet2)
            let coreMLModel = try FoodMainDishEfficientNet(configuration: config).model
            self.visionModel = try VNCoreMLModel(for: coreMLModel)
            print("✅ Main dish model loaded OK")
        } catch {
            self.visionModel = nil
            print("❌ Failed to load main dish model:", error)
        }
    }
    
    // MARK: - Public APIs
    
    /// Dùng cho realtime camera (CVPixelBuffer)
    func classify(pixelBuffer: CVPixelBuffer) {
        // throttle
        let now = CFAbsoluteTimeGetCurrent()
        guard now - lastUpdateTime >= updateInterval else { return }
        lastUpdateTime = now
        
        performClassification {
            VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        }
    }
    
    /// Dùng cho ảnh tĩnh upload (CGImage)
    func classify(cgImage: CGImage) {
        performClassification {
            VNImageRequestHandler(cgImage: cgImage, options: [:])
        }
    }
    
    // MARK: - Private helper
    
    private func performClassification(makeHandler: () -> VNImageRequestHandler) {
        guard let visionModel = visionModel else { return }
        
        let request = VNCoreMLRequest(model: visionModel) { [weak self] request, error in
            guard let self else { return }
            
            if let error {
                print("❌ VNCoreMLRequest error:", error)
                return
            }
            
            guard let results = request.results as? [VNClassificationObservation],
                  let best = results.first else {
                self.mainDishLabel = nil
                self.mainDishConfidence = 0
                return
            }
            
            if best.confidence >= self.minDishConfidence {
                self.mainDishLabel = best.identifier
                self.mainDishConfidence = best.confidence
                print("🍜 Main dish:", best.identifier, best.confidence)
            } else {
                // Model không đủ tự tin → không show món chính
                self.mainDishLabel = nil
                self.mainDishConfidence = 0
                print("🤷‍♂️ Main dish too uncertain:", best.identifier, best.confidence)
            }
        }
        
        request.imageCropAndScaleOption = .centerCrop
        
        let handler = makeHandler()
        try? handler.perform([request])
    }
}

