using System;
using Microsoft.ML.Models;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;


namespace SentimentAnalysis
{
    class Program
    {
        const string _dataPath = @"data/imdb_labelled.txt";



        static void Main(string[] args)
        {
          var model = TrainAndPredict();

        }

        public static PredictionModel<SentimentData, SentimentPrediction> TrainAndPredict()
        {
            var pipeline = new LearningPipeline();
            pipeline.Add(new TextLoader<SentimentData>(_dataPath,useHeader:false,separator:"tab"));
            pipeline.Add(new TextFeaturizer("Features", "SentimentText"));
            pipeline.Add(new FastTreeBinaryClassifier() { NumLeaves = 5, NumTrees = 5, MinDocumentsInLeafs = 2 });


            IEnumerable<SentimentData> sentiments = new[]
               {
                new SentimentData
                {
                    SentimentText = "Contoso's 11 is a wonderful experience",
                    Sentiment = 0
                },
                new SentimentData
                {
                    SentimentText = "The acting in this movie is very bad",
                    Sentiment = 0
                },
                new SentimentData
                {
                    SentimentText = "Joe versus the Volcano Coffee Company is a great film.",
                    Sentiment = 0
                }
            };


            PredictionModel<SentimentData,SentimentPrediction> model =  pipeline.Train<SentimentData, SentimentPrediction>();

            IEnumerable<SentimentPrediction> _predictions = model.Predict(sentiments);
           


            Console.WriteLine();
            Console.WriteLine("Sentiment Predictions");
            Console.WriteLine("---------------------");

            var sentimentsAndPredictions = sentiments.Zip(_predictions, (sentiment, prediction) => (sentiment, prediction));

            foreach (var item in sentimentsAndPredictions)
            {
                Console.WriteLine($"Sentiment: {item.sentiment.SentimentText} | Prediction: {(item.prediction.Sentiment ? "Positive" : "Negative")}");
            }
            Console.WriteLine();


            return model;


        }
    }
}
