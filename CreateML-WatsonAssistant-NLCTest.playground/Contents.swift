import Foundation

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
        let dataset = try JSONDecoder().decode(Dataset.self, from: data)


        print(dataset)

    }
    catch let error {
        print("\(error.localizedDescription)")
    }
}

