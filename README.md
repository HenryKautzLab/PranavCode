This branch is the latest one yet, here I have used my existing LLaVA code but this time added OpenAI Whisper and easyOCR for even better analysis. I used OpenAI Whisper to gather audio, and easyOCR to gather text, then I fed this into LLavA on top of the original questions. This gave LLaVA more context about what is going on in the video, especially since LLaVA doesn't support audio and can sometimes miss out on text in the video. 

Whisper proved especially useful in one video about a news report, where audio is very important. 

EasyOCR gives better context in case the video has subittles or other meaningful text on the screen.


03/09:
Used all-MiniLM-L6-v2 to measure similarity between 1 model and 3 model outputs.

I found that using 3 models made the similarity very moderate, in the 0.6 range which makes sense as more details are added to the output. I have done this for 3 videos which may not be a lot due to power limitation but am confident that because these 3 videos had similarity scores very similar to ecah other, the other videos would exhibit the same behavior.

Next, I will try to do more comparisons for the rest of the videos.
