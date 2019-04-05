The file contains YouTube comments that are labelled as either 'counterspeech' or 'non-counterspeech'. Each comment is in the form of key-value pair. The details of each such comment are as follows:
~~~
Community             -->  The community which was targetted in the YouTube Video
CounterSpeech         -->  This field indicates if the comment is a counterspeech or not.
Category              -->  The type of counterspeech used in the comment text. The possible values are : 1 - Presentation of fact, 2 - Pointing out hypocrisy or contradiction, 3 - Warning of offline or online consequences, 4 - Affiliation, 5 - Denouncing hateful or dangerous speech, 7 - Humor, 8 - Positive Tone, 9 - Hostile Language, Default - Non-Counterspeech.
commentText           -->  The comment tagged by the annotator.
hasReplies            -->  A boolean field indicating if the comments has received any replies or not.
id                    -->  The unique id assigned to each comment.
likes                 -->  The number of likes received by the comment.
numberOfReplies       -->  The number of replies received by the comment.
replies               -->  The list of replies received by the comment.
timestamp             -->  The timestamp of the comment
user                  -->  The account name of the user who posted the comment.
~~~

You can use json to load the dataset:

```
import json
dataset = json.load('Counterspeech_Dataset.json')
```
