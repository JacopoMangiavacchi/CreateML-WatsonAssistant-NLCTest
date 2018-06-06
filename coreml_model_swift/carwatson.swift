//
// carwatson.swift
//
// This file was automatically generated and should not be edited.
//

import CoreML


/// Model Prediction Input Type
@available(macOS 10.14, iOS 12.0, tvOS 12.0, watchOS 5.0, *)
class carwatsonInput : MLFeatureProvider {

    /// Input text as string value
    var text: String

    var featureNames: Set<String> {
        get {
            return ["text"]
        }
    }
    
    func featureValue(for featureName: String) -> MLFeatureValue? {
        if (featureName == "text") {
            return MLFeatureValue(string: text)
        }
        return nil
    }
    
    init(text: String) {
        self.text = text
    }
}

/// Model Prediction Output Type
@available(macOS 10.14, iOS 12.0, tvOS 12.0, watchOS 5.0, *)
class carwatsonOutput : MLFeatureProvider {

    /// Source provided by CoreML

    private let provider : MLFeatureProvider


    /// Text label as string value
    lazy var label: String = {
        [unowned self] in return self.provider.featureValue(for: "label")!.stringValue
    }()

    var featureNames: Set<String> {
        return self.provider.featureNames
    }
    
    func featureValue(for featureName: String) -> MLFeatureValue? {
        return self.provider.featureValue(for: featureName)
    }

    init(label: String) {
        self.provider = try! MLDictionaryFeatureProvider(dictionary: ["label" : MLFeatureValue(string: label)])
    }

    init(features: MLFeatureProvider) {
        self.provider = features
    }
}


/// Class for model loading and prediction
@available(macOS 10.14, iOS 12.0, tvOS 12.0, watchOS 5.0, *)
class carwatson {
    var model: MLModel

    /**
        Construct a model with explicit path to mlmodel file
        - parameters:
           - url: the file url of the model
           - throws: an NSError object that describes the problem
    */
    init(contentsOf url: URL) throws {
        self.model = try MLModel(contentsOf: url)
    }

    /// Construct a model that automatically loads the model from the app's bundle
    convenience init() {
        let bundle = Bundle(for: carwatson.self)
        let assetPath = bundle.url(forResource: "carwatson", withExtension:"mlmodelc")
        try! self.init(contentsOf: assetPath!)
    }

    /**
        Make a prediction using the structured interface
        - parameters:
           - input: the input to the prediction as carwatsonInput
        - throws: an NSError object that describes the problem
        - returns: the result of the prediction as carwatsonOutput
    */
    func prediction(input: carwatsonInput) throws -> carwatsonOutput {
        return try self.prediction(input: input, options: MLPredictionOptions())
    }

    /**
        Make a prediction using the structured interface
        - parameters:
           - input: the input to the prediction as carwatsonInput
           - options: prediction options 
        - throws: an NSError object that describes the problem
        - returns: the result of the prediction as carwatsonOutput
    */
    func prediction(input: carwatsonInput, options: MLPredictionOptions) throws -> carwatsonOutput {
        let outFeatures = try model.prediction(from: input, options:options)
        return carwatsonOutput(features: outFeatures)
    }

    /**
        Make a prediction using the convenience interface
        - parameters:
            - text: Input text as string value
        - throws: an NSError object that describes the problem
        - returns: the result of the prediction as carwatsonOutput
    */
    func prediction(text: String) throws -> carwatsonOutput {
        let input_ = carwatsonInput(text: text)
        return try self.prediction(input: input_)
    }

    /**
        Make a batch prediction using the structured interface
        - parameters:
           - inputs: the inputs to the prediction as [carwatsonInput]
           - options: prediction options 
        - throws: an NSError object that describes the problem
        - returns: the result of the prediction as [carwatsonOutput]
    */
    func predictions(inputs: [carwatsonInput], options: MLPredictionOptions) throws -> [carwatsonOutput] {
        if #available(macOS 10.14, iOS 12.0, tvOS 12.0, watchOS 5.0, *) {
            let batchIn = MLArrayBatchProvider(array: inputs)
            let batchOut = try model.predictions(from: batchIn, options: options)
            var results : [carwatsonOutput] = []
            results.reserveCapacity(inputs.count)
            for i in 0..<batchOut.count {
                let outProvider = batchOut.features(at: i)
                let result =  carwatsonOutput(features: outProvider)
                results.append(result)
            }
            return results
        } else {
            var results : [carwatsonOutput] = []
            results.reserveCapacity(inputs.count)
            for input in inputs {
                let result = try self.prediction(input: input, options: options)
                results.append(result)
            }
            return results
        }
    }
}
