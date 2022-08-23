//
//  ContentView.swift
//  Augmented Labels
//
//  Created by Bill Moriarty on 8/22/22.
//

import SwiftUI
import RealityKit
import CoreML
import Vision


class ModelRecognizer: ObservableObject {
    private init() { }
    
    static let shared = ModelRecognizer()
    
    @Published var aView = ARView()
    @Published var recognizedObject = "nothing yet"
    @Published var model = try! VNCoreMLModel(for: Resnet50().model) //Resnet50().model)
    
    var timer = Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true, block: { _ in
        continuouslyUpdate()
    })
    
    func setRecognizedObject(newThing: String){
        recognizedObject = newThing
    }
    
}


struct ContentView : View {
    
    var body: some View {
        return WrappingView()
    }
}


struct WrappingView: View {
    @ObservedObject var recogd: ModelRecognizer = .shared
    
    var body: some View {
        ZStack{
            
            ARViewContainer().edgesIgnoringSafeArea(.all)
            
            Text("Score: \(recogd.recognizedObject)")
            
        }
    }
}


struct ExampleView: View {
    @Binding var message: String
    
    var body: some View {
        Text(message).font(.title)
    }
}


struct ARViewContainer: UIViewRepresentable {
    
    
    @ObservedObject var recogd: ModelRecognizer = .shared
    
    //    var arView = ARView(frame: .zero)
    
    
    
    
    func makeUIView(context: Context) -> ARView {
        
        
        // Load the "Box" scene from the "Experience" Reality File
        let boxAnchor = try! Experience.loadBox()
        
        // Add the box anchor to the scene
        var v = recogd.aView
        v.scene.anchors.append(boxAnchor)
        
        return v
        
    }
    
    func updateUIView(_ uiView: ARView, context: Context) {
        
    }
    
    
}

func continuouslyUpdate() {
    print("continuouslyUpdate")
    
    @ObservedObject var recogd: ModelRecognizer = .shared
    
    let v = recogd.aView
    let sess = v.session
    let mod = recogd.model
    

    let tempImage: CVPixelBuffer? = sess.currentFrame?.capturedImage
    
    var firstResult = String ("")
    //get the current camera frame from the live AR session
    if tempImage == nil { return }
    let tempciImage = CIImage(cvPixelBuffer: tempImage!)
    
    //initiate the request
    let request = VNCoreMLRequest(model: mod) { (request, error) in }
    //crop just the center of the captured camera frame to send to the ML model
    request.imageCropAndScaleOption=VNImageCropAndScaleOption.centerCrop
    
    let handler = VNImageRequestHandler(ciImage: tempciImage)
    do {
        //send the request to the model
        try handler.perform([request])
    } catch {
        print(error)
    }
    
    //process the result of the request
    guard let results = request.results as? [VNClassificationObservation] else {
        fatalError("model failed to process image")
    }
    //format the result into a string
    firstResult = results.first.flatMap({ $0 as VNClassificationObservation })
        .map({ "\($0.identifier) \(String(format:"- %.2f", $0.confidence))" })!
    
    DispatchQueue.main.async {
        recogd.setRecognizedObject(newThing: firstResult)
    }
    
    print(firstResult)
    
}

#if DEBUG
struct ContentView_Previews : PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
#endif
