//
//  Detector.swift
//  YOLOv11-Object-Detector-iOS-CoreML
//

import CoreML
import AVFoundation
import Vision
import CoreGraphics

@Observable @MainActor
final class Detector {
    private let visionModel: VNCoreMLModel?
    
    // Kết quả detect để vẽ box
    var detectedObjects: [BoundingBox] = []
    // Thời gian xử lý 1 frame (ms)
    var interfaceTime: Double = 0.0
    
    var pixelsWide = 0
    var pixelsHigh = 0
    var labels: [String] = []
    
    // Ngưỡng confidence YOLO
    let confidenceThreshold = 0.35
    
    // ====== Smoothing ======
    private var previousDetections: [BoundingBox] = []
    private let smoothingAlpha: CGFloat = 0.8       // 0.8 = bám frame cũ mạnh hơn
    private var lastUpdateTime: CFAbsoluteTime = 0
    private let updateInterval: CFAbsoluteTime = 0.20   // 0.2s ~ 5 FPS cho đỡ rung
    
    init(modelName: String) {
        do {
            self.visionModel = try Utils.loadModel(named: modelName)
        } catch {
            self.visionModel = nil
            fatalError("❌ Failed to load model: \(error)")
        }
    }
    
    // MARK: - Detect realtime từ camera (như cũ, nhưng có smoothing)
    func detectObjects(pixelBuffer: CVPixelBuffer) {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        guard let visionModel = self.visionModel else { return }
        
        // Throttle: tránh chạy quá dày
        let now = CFAbsoluteTimeGetCurrent()
        guard now - lastUpdateTime >= updateInterval else { return }
        lastUpdateTime = now
        
        let request = VNCoreMLRequest(model: visionModel) { [weak self] request, error in
            guard let self = self else { return }
            
            guard let results = request.results as? [VNRecognizedObjectObservation] else {
                DispatchQueue.main.async {
                    self.detectedObjects = []
                    self.interfaceTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000.0
                }
                return
            }
            
            let rawDetections: [BoundingBox] = results.compactMap { observation in
                let confidence = observation.confidence
                guard Double(confidence) >= self.confidenceThreshold else { return nil }
                guard let topLabel = observation.labels.first else { return nil }
                
                return BoundingBox(
                    label: topLabel.identifier,
                    boundingBox: observation.boundingBox,
                    confidence: confidence
                )
            }
            
            let smoothed = self.smoothDetections(rawDetections)
            
            DispatchQueue.main.async {
                self.detectedObjects = smoothed
                self.interfaceTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000.0
            }
        }
        
        request.imageCropAndScaleOption = VNImageCropAndScaleOption.scaleFit
        
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        try? handler.perform([request])
    }
    
    // MARK: - Detect trên ảnh tĩnh (dùng cho "Nhận diện qua hình")
    func detectObjects(on cgImage: CGImage) {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        guard let visionModel = self.visionModel else { return }
        
        let request = VNCoreMLRequest(model: visionModel) { [weak self] request, error in
            guard let self = self else { return }
            
            guard let results = request.results as? [VNRecognizedObjectObservation] else {
                DispatchQueue.main.async {
                    self.detectedObjects = []
                    self.interfaceTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000.0
                }
                return
            }
            
            let rawDetections: [BoundingBox] = results.compactMap { observation in
                let confidence = observation.confidence
                guard Double(confidence) >= self.confidenceThreshold else { return nil }
                guard let topLabel = observation.labels.first else { return nil }
                
                return BoundingBox(
                    label: topLabel.identifier,
                    boundingBox: observation.boundingBox,
                    confidence: confidence
                )
            }
            
            // Ảnh tĩnh có thể dùng lại smoothing cho đồng nhất,
            // hoặc bỏ smoothing nếu muốn box "thô" — tạm reuse luôn.
            let smoothed = self.smoothDetections(rawDetections)
            
            DispatchQueue.main.async {
                self.detectedObjects = smoothed
                self.interfaceTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000.0
            }
        }
        
        request.imageCropAndScaleOption = VNImageCropAndScaleOption.scaleFit
        
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        try? handler.perform([request])
    }
    
    // MARK: - Smoothing helpers
    
    private func smoothDetections(_ current: [BoundingBox]) -> [BoundingBox] {
        // Frame đầu tiên thì chưa có gì để smooth
        guard !previousDetections.isEmpty else {
            previousDetections = current
            return current
        }
        
        var usedPrevious = Set<Int>()
        var output: [BoundingBox] = []
        
        for curr in current {
            var bestIndex: Int?
            var bestIoU: CGFloat = 0
            
            for (index, prev) in previousDetections.enumerated() where !usedPrevious.contains(index) {
                let iouValue = iou(curr.boundingBox, prev.boundingBox)
                if iouValue > bestIoU {
                    bestIoU = iouValue
                    bestIndex = index
                }
            }
            
            // Nếu trùng với box cũ (IoU đủ lớn) thì blend lại
            if let idx = bestIndex, bestIoU > 0.3 {
                usedPrevious.insert(idx)
                let prev = previousDetections[idx]
                let blendedRect = blend(prev.boundingBox, curr.boundingBox, alpha: smoothingAlpha)
                let blendedConf = smoothingAlpha * CGFloat(prev.confidence) +
                                  (1 - smoothingAlpha) * CGFloat(curr.confidence)
                
                output.append(
                    BoundingBox(
                        label: curr.label,
                        boundingBox: blendedRect,
                        confidence: Float(blendedConf)
                    )
                )
            } else {
                // Box mới → lấy nguyên
                output.append(curr)
            }
        }
        
        previousDetections = output
        return output
    }
    
    private func blend(_ a: CGRect, _ b: CGRect, alpha: CGFloat) -> CGRect {
        let beta = 1 - alpha
        return CGRect(
            x: a.origin.x * alpha + b.origin.x * beta,
            y: a.origin.y * alpha + b.origin.y * beta,
            width: a.size.width * alpha + b.size.width * beta,
            height: a.size.height * alpha + b.size.height * beta
        )
    }
    
    private func iou(_ a: CGRect, _ b: CGRect) -> CGFloat {
        let intersection = a.intersection(b)
        if intersection.isNull { return 0 }
        
        let interArea = intersection.width * intersection.height
        let unionArea = a.width * a.height + b.width * b.height - interArea
        if unionArea <= 0 { return 0 }
        return interArea / unionArea
    }
}

