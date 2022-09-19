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
import SceneKit
import ARKit

// create and observable object that structs can access
class ModelRecognizer: ObservableObject {
    private init() { }
    
    static let shared = ModelRecognizer()
    
    @Published var aView = ARView()
    @Published var recognizedObject = "nothing yet"
    
    // using ! because the model is required and it is in my XCode project
    @Published var model = try! VNCoreMLModel(for: Resnet50().model) //Resnet50().model)
    
    // call the continuouslyUpdate function every half second
    var timer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true, block: { _ in
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
        }
    }
}


struct ARViewContainer: UIViewRepresentable {
    
    @ObservedObject var recogd: ModelRecognizer = .shared
    
    func makeUIView(context: Context) -> ARView {
        let v = recogd.aView
        return v
        
    }
    
    func updateUIView(_ uiView: ARView, context: Context) {
        var txt = SCNText()
        
        // let's keep the number of anchors to no more than 5 just so the screen doesn't get cluttered
        if recogd.aView.scene.anchors.count > 5 {
            recogd.aView.scene.anchors.remove(at: 0)
        }
        
        // create the AR Text to place on the screen
        txt = SCNText(string: recogd.recognizedObject, extrusionDepth: 2)
        let material = SCNMaterial()
        material.diffuse.contents = UIColor.magenta
        txt.materials = [material]
        
        let randomColor = generateRandomColor()
        
        let shader = SimpleMaterial(color: randomColor, roughness: 1, isMetallic: true)
        let text = MeshResource.generateText(
            "\(recogd.recognizedObject)",
            extrusionDepth: 0.08,
            font: .systemFont(ofSize: 0.1, weight: .bold),
            alignment: .center
        )
        
        let textEntity = ModelEntity(mesh: text, materials: [shader])
        
        let transform = recogd.aView.cameraTransform
        
        // set the transform (the 3d location) of the text to be near the center of the camer, and 1 meter away
        let trans2 = simd_float4x4(transform.matrix)
        let anchEntity = AnchorEntity(world: trans2)
        textEntity.position.z -= 1.0
        
        anchEntity.addChild(textEntity)
        
        // add this anchor entity to the scene
        recogd.aView.scene.addAnchor(anchEntity)
    }
    
}

func generateRandomColor() -> UIColor {
    let redValue = CGFloat(drand48())
    let greenValue = CGFloat(drand48())
    let blueValue = CGFloat(drand48())
    
    let randomColor = UIColor(red: redValue, green: greenValue, blue: blueValue, alpha: 1.0)
    
    return randomColor
}

func continuouslyUpdate() {
    
    @ObservedObject var recogd: ModelRecognizer = .shared
    
    // access what we need
    let v = recogd.aView
    let sess = v.session
    let mod = recogd.model
    
    // access the current frame as an image
    let tempImage: CVPixelBuffer? = sess.currentFrame?.capturedImage
    
    var firstResult = String ("")
    
    //get the current camera frame from the live AR session
    if tempImage == nil { return }
    let tempciImage = CIImage(cvPixelBuffer: tempImage!)
    
    // create a reqeust to the Vision Core ML Model
    let request = VNCoreMLRequest(model: mod) { (request, error) in }
    //crop just the center of the captured camera frame to send to the ML model
    request.imageCropAndScaleOption=VNImageCropAndScaleOption.centerCrop
    
    // perform the request
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
        .map({ "\($0.identifier)" })!
    
    // parse before comma
    if let index = firstResult.firstIndex(of: ",") {
        firstResult = String(firstResult.prefix(upTo: index))
    }
    
    // if we found something new, set the recognized object
    if !recogd.recognizedObject.elementsEqual(firstResult) {
        DispatchQueue.main.async {
            recogd.setRecognizedObject(newThing: firstResult)
        }
    }
    
}

#if DEBUG
struct ContentView_Previews : PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
#endif
