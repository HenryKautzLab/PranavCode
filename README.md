This brnach is the latest one yet, here I have used my existing LLaVA code but this time added OpenAI Whisper and easyOCR for even better analysis. I used OpenAI Whisper to gather audio, and easyOCR to gather text, then I fed this into LLavA on top of the original questions. This gave LLaVA more context about what is going on in the video, especially since LLaVA doesn't support audio and can sometimes miss out on text in the video. 

Whisper proved especially useful in one video about a news report, where audio is very important. 

EasyOCR gives better context in case the video has subittles or other meaningful text on the screen.
