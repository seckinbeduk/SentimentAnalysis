# SentimentAnalysis

The sample is a console app that uses the ML.NET API to train a model that classifies and predicts sentiment as either positive or negative

# Test data for prediction

```csharp
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
            
            
```


#Result

Sentiment Predictions </br>
---------------------  </br>
Sentiment: Contoso's 11 is a wonderful experience | Prediction: Positive </br>
Sentiment: The acting in this movie is very bad | Prediction: Negative </br>
Sentiment: Joe versus the Volcano Coffee Company is a great film. | Prediction: Positive </br> 
