//
//  ContentView.swift
//  Augmented Labels
//
//  Created by Bill Moriarty on 10/12/22.
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
    
    // instantiate the core ML model
    // I am using ! here because the model is required for this application to function q
    @Published var model  = try! VNCoreMLModel(for: MobileNetV2().model)
    
    // call the continuouslyUpdate function every half second
    var timer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true, block: { _ in
        continuouslyUpdate()
    })
    
    func setRecognizedObject(newThing: String){
        recognizedObject = newThing
    }
}

struct ContentView : View {
    
    var body: some View {
        VStack {
            Text(verbatim: "Hold Your Phone In Portrait Mode")
            WrappingView()
        }
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
        
        // let's keep the number of anchors to no more than 1 for this demo
        if recogd.aView.scene.anchors.count > 0 {
            recogd.aView.scene.anchors.removeAll()
        }
        
        // create the AR Text to place on the screen
        txt = SCNText(string: recogd.recognizedObject, extrusionDepth: 1)
        let material = SCNMaterial()
        material.diffuse.contents = UIColor.magenta
        txt.materials = [material]
        
        let randomColor = generateRandomColor()
        
        let shader = SimpleMaterial(color: randomColor, roughness: 1, isMetallic: true)
        let text = MeshResource.generateText(
            "\(recogd.recognizedObject)",
            extrusionDepth: 0.05,
            font: .init(name: "Helvetica", size: 0.05)!,
            alignment: .center
        )
        
        let textEntity = ModelEntity(mesh: text, materials: [shader])
        
        let transform = recogd.aView.cameraTransform
        
        // set the transform (the 3d location) of the text to be near the center of the camera, and 1/2 meter away
        let trans = simd_float4x4(transform.matrix)
        let anchEntity = AnchorEntity(world: trans)
        textEntity.position.z -= 0.5 // place the text 1/2 meter away from the camera along the Z axis
        
        // find the width of the entity in order to have the text appear in the center
        let minX = text.bounds.min.x
        let maxX = text.bounds.max.x
        let width = maxX - minX
        let xPos = width / 2
        
        textEntity.position.x = transform.translation.x - xPos
        
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
    
    // access what we need from the observed object
    let v = recogd.aView
    let sess = v.session
    let mod = recogd.model
    
    // access the current frame as an image
    let tempImage: CVPixelBuffer? = sess.currentFrame?.capturedImage
    
    
    //get the current camera frame from the live AR session
    if tempImage == nil {
        return
    }
    
    let tempciImage = CIImage(cvPixelBuffer: tempImage!)
    
    // create a reqeust to the Vision Core ML Model
    let request = VNCoreMLRequest(model: mod) { (request, error) in }
    
    //crop just the center of the captured camera frame to send to the ML model
    request.imageCropAndScaleOption = .centerCrop
    
    // perform the request
    let handler = VNImageRequestHandler(ciImage: tempciImage, orientation: .down) //left //right
    
    do {
        //send the request to the model
        try handler.perform([request])
    } catch {
        print(error)
    }
    
    guard let observations = request.results as? [VNClassificationObservation] else { return}
    
    // only proceed if the model prediction's confidence in the first result is greater than 50%
    if observations[0].confidence < 0.5  { return }
    
    // the model returns predictions in descending order of confidence
    // we want to select the first prediction, which has the higest confidence
    let topLabelObservation = observations[0].identifier
    
    // get just the first word from the prediction string
    let firstWord = topLabelObservation.components(separatedBy: [","])[0]
    
    if recogd.recognizedObject != firstWord {
        DispatchQueue.main.async {
            recogd.setRecognizedObject(newThing: firstWord)
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
