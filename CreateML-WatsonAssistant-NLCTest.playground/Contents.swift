import Foundation
import CreateML

public struct Intent: Codable {
    let intent: String
    let utterances: [String]
}

public struct Dataset : Codable {
    let intents: [Intent]
}

if let fileURL = Bundle.main.url(forResource: "dataset", withExtension: "json") {
    do {
        let data = try Data(contentsOf: fileURL)
        let dataset: Dataset = try JSONDecoder().decode(Dataset.self, from: data)

        var dictionary = [String : MLDataValueConvertible]()
        var textArray = [String]()
        var labelArray = [String]()

        for intent in dataset.intents {
            for utterance in intent.utterances {
                labelArray.append(intent.intent)
                textArray.append(utterance)
            }
        }

        dictionary["text"] = textArray
        dictionary["label"] = labelArray

        let dataTable = try MLDataTable(dictionary: dictionary)
        
        let (trainingData, testingData) = dataTable.randomSplit(by: 0.8, seed: 5)
        
        let watsonClassifier = try MLTextClassifier(trainingData: trainingData,
                                                       textColumn: "text",
                                                       labelColumn: "label")
        
        let trainingAccuracy = (1.0 - watsonClassifier.trainingMetrics.classificationError) * 100
        let validationAccuracy = (1.0 - watsonClassifier.validationMetrics.classificationError) * 100
        let evaluationMetrics = watsonClassifier.evaluation(on: testingData)
        let evaluationAccuracy = (1.0 - evaluationMetrics.classificationError) * 100
        

        print(trainingAccuracy, validationAccuracy, evaluationMetrics, evaluationAccuracy)

        let metadata = MLModelMetadata(author: "Jacopo Mangiavacchi",
                                       shortDescription: "Watson Assistant Car demo intent classifier",
                                       version: "1.0")
        
        try watsonClassifier.write(to: URL(fileURLWithPath: "CarWatson.mlmodel"), metadata: metadata)
    }
    catch let error {
        print("\(error.localizedDescription)")
    }
}

