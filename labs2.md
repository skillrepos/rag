# Gen AI: Understanding and Using RAG
## Making LLMs smarter by pairing your data with Gen AI
## Session labs 
## Revision 1.0 - 09/16/24

**Follow the startup instructions in the README.md file IF NOT ALREADY DONE!**

**NOTE: To copy and paste in the codespace, you may need to use keyboard commands - CTRL-C and CTRL-V. Chrome may work best for this.**

**Lab 1 - Implementing basic RAG with prompt context**

**Purpose: In this lab, we’ll see a basic example of RAG by retrieving prompt context from a data file.**

1. First, download and install Ollama to allow us to have a local LLM to query. Go to the *TERMINAL* tab in the bottom part of the codespace and execute the command below. Then you can run the actual application to see usage.
```
curl -fsSL https://ollama.com/install.sh | sh
```
![downloading ollama](./images/rag05.png?raw=true "downloading ollama")

2. Next, start the ollama server running and pull down the *llama3* model with the following command:
```
ollama serve && ollama pull llama3 &
```
![serve and pull](./images/rag06a.png?raw=true "serve and pull")

3. While this is running, go ahead and open a second terminal session. In the codespace, right-click and select the *Split Terminal* option. This will add a second terminal side-by-side with your other one.

![splitting terminal](./images/rag08.png?raw=true "splitting terminal")
   
4. In our repository, we have a set of Python programs to help us illustrate and work with concepts in the labs. These are mostly in the *genai* subdirectory. In a free terminal session, change into that directory. (Remaining steps will assume that directory.)
```
cd genai
```

5. For this lab, we have a simple Pyton program that queries the LLM about the benefits of Python in general. The file name is lab1.py. Open the file either by clicking on [**genai/lab1.py**](./genai/lab1.py) or by entering the command below in the codespace's terminal.

```
code lab1.py
```

5. You can look around this file to see how it works. Notice the prompt "Explain the benefits of Python" on line 17. When you are done, and after step 2 has completed, go ahead and run the file with the command below. 

```
python lab1.py
```
![running lab1 file](./images/rag09.png?raw=true "running lab1 file")

6. This will take several minutes to run. When it's done, you'll be able to see a rather large amount of output about the benefits of python.

![lab output 1](./images/rag10.png?raw=true "lab output 1")
  
7. Now, let's create a copy of the program that adds context from a local *knowledge base*. For purposes of keeping things simple and quick, we have a local file that we'll use for the additional context. The file name is *data/kb.txt*. Open the file either by clicking on [**data/kb.txt**](./data/kb.txt) or by entering the command below in the codespace's terminal.  

```
code ../data/kb.txt
```

8. Now let's update the python file to pass the context from the file. Switch back to the *lab1.py* file. First, after the imports, add the code below to read in the data file.

```
# Read the text from the file
with open("../data/kb.txt", "r") as file:
    knowledge_data = file.read()
```
![mod 1](./images/rag11.png?raw=true "mod 1")

9. Now scroll down to where the prompt is defined (probably around line 20) and modify that line to use the prompt below. Notice that it adds on the "{knowledge_data}" section, which is the content we read in. It also sets context for the LLM "Based on the following information" which refers to the knowledge base again.

```
    "prompt": f"Based on the following information, explain the benefits of Python:\n\n{knowledge_data}",
```
![mod 2](./images/rag12.png?raw=true "mod 2")

10. Save your changes to the file with CMD+S or CTRL+S (or use the three bar menu in the top left of the codespace and drill-down to the File->Save option). Then run the program again.

```
(save file)
python lab1.py
```

11. After the run is done, you should see output that indicates the AI referenced the data in the file to help answer the prompt.

![run 2](./images/rag13.png?raw=true "run 2") 


<p align="center">
**[END OF LAB]**
</p>
</br></br>

**Lab 2 - Querying a Knowledge Base with Basic RAG**

**Purpose: In this lab, we'll see how vectorize and query data from a knowledge base that we can then have the LLM expand on.**

1. For this lab, in the same *genai* directory, we have a simple Python program to read a knowledge base and store it in a vectorized format that we can then query against for matches. The file name is lab2.py. Open the file either by clicking on [**genai/lab2.py**](./genai/lab2.py) or by entering the command below in the codespace's terminal.

```
code lab2.py
```

2. Since we need a *knowledge base* to work with, we have a local file that we'll use. The file name is *data/kb.json*. Open the file either by clicking on [**data/kb.json**](./data/kb.json) or by entering the command below in the codespace's terminal.  

```
code ../data/kb.json
```

2. This program can be run and passed a model to use for tokenization. To start, we'll be using a model named *bert-base-uncased*. Let's look at this model on huggingface.co.  Go to https://huggingface.co/models and in the *Models* search area, type in *bert-base-uncased*. Select the entry for *google-bert/bert-base-uncased*.

![Finding bert model on huggingface](./images/gaidd12.png?raw=true "Finding bert model on huggingface")

3. Once you click on the selection, you'll be on the *model card* tab for the model. Take a look at the model card for the model and then click on the *Files and Versions* and *Community* tabs to look at those pages.

![huggingface tabs](./images/gaidd13.png?raw=true "huggingface tabs")

4. Now let's switch back to the codespace and, in the terminal, run the *tokenizer* program with the *bert-base-uncased* model. Enter the command below. This will download some of the files you saw on the *Files* tab for the model in HuggingFace.
```
python tokenizer.py bert-base-uncased
```
5. After the program starts, you will be at a prompt to *Enter text*. Enter in some text like the following to see how it will be tokenized.
```
This is sample text for tokenization and text for embeddings.
```
![input for tokenization](./images/gaidd36.png?raw=true "input for tokenization")

6. After you enter this, you'll see the various subword tokens that were extracted from the text you entered. And you'll also see the ids for the tokens stored in the model that matched the subwords.

![tokenization output](./images/gaidd37.png?raw=true "tokenization output")

7. Next, you can try out some other models. Repeat steps 4 - 6 for other tokenizers like the following. (You can use the same text string or different ones. Notice how the text is broken down depending on the model and also the meta-characters.)
```
python tokenizer.py roberta-base
python tokenizer.py gpt2
python tokenizer.py xlnet-large-cased
```
8. (Optional) If you finish early and want more to do, you can look up the models from step 7 on huggingface.co/models.
   
<p align="center">
**[END OF LAB]**
</p>
</br></br>

**Lab 3 - Understanding embeddings, vectors and similarity measures**

**Purpose: In this lab, we'll see how tokens get mapped to vectors and how vectors can be compared.**

1. In the repository, we have a Python program that uses a Tokenizer and Model to create embeddings for three terms that you input. It then computes and displays the cosine similarity between each combination. Open the file to look at it by clicking on [**genai/vectors.py**](./genai/vectors.py) or by using the command below in the terminal.
```
code vectors.py
```
2. Let's run the program. As we did for the tokenizer example, we'll pass in a model to use. We'll also pass in a second argument which is the number of dimensions from the vector for each term to show. Run the program with the command below. You can wait to enter terms until the next step.
```
python vectors.py bert-base-cased 5
```
![vectors program run](./images/gaidd38.png?raw=true "vectors program run")

3. The command we just ran loads up the bert-base-cased model and tells it to show the first 5 dimensions of each vector for the terms we enter. The program will be prompting you for three terms. Enter each one in turn. You can try two closely related words and one that is not closely related. For example
   - king
   - queen
   - duck

![vectors program inputs](./images/gaidd39.png?raw=true "vectors program inputs")

4. Once you enter the terms, you'll see the first 5 dimensions for each term. And then you'll see the cosine similarity displayed between each possible pair. This is how similar each pair of words is. The two that are most similar should have a higher cosine similarity "score".

![vectors program outputs](./images/gaidd40.png?raw=true "vectors program outputs")

5. Each vector in the bert-based models have 768 dimensions. Let's run the program again and tell it to display 768 dimensions for each of the three terms.  Also, you can try another set of terms that are more closely related, like *multiplication*, *division*, *addition*.
```
python vectors.py bert-base-cased 768
```
6. You should see that the cosine similarities for all pair combinations are not as far apart this time.
![vectors program second outputs](./images/gaidd19.png?raw=true "vectors program second outputs")

7. As part of the output from the program, you'll also see the *token id* for each term. (It is above the print of the dimensions. If you don't want to scroll through all the dimensions, you can just run it again with a small number of dimensions like we did in step 2.) If you're using the same model as you did in lab 2 for tokenization, the ids will be the same. 

![token id](./images/gaidd20.png?raw=true "token id")

8. You can actually see where these mappings are stored if you look at the model on Hugging Face. For instance, for the *bert-base-cased* model, you can go to https://huggingface.co and search for bert-base-cased. Select the entry for google-bert/bert-base-cased.

![finding model](./images/gaidd21.png?raw=true "finding model")

8. On the page for the model, click on the *Files and versions* tab. Then find the file *tokenizer.json* and click on it. The file will be too large to display, so click on the *check the raw version* link to see the actual content.

![selecting tokenizer.json](./images/gaidd22.png?raw=true "selecting tokenizer.json")
![opening file](./images/gaidd23.png?raw=true "opening file")

9. You can search for the terms you entered previously with a Ctrl-F or Cmd-F and find the mapping between the term and the id. If you look for "##" you'll see mappings for parts of tokens like you may have seen in lab 2.

![finding terms in file](./images/gaidd24.png?raw=true "finding terms in files")

10. If you want, you can try running the *genai_vectors.py* program with a different model to see results from other models (such as we used in lab 2) and words that are very close like *embeddings*, *tokenization*, *subwords*.

<p align="center">
**[END OF LAB]**
</p>
</br></br>

**Lab 4 - Working with transformer models**

**Purpose: In this lab, we’ll see how to interact with various models for different standard tasks**

1. In our repository, we have several different Python programs that utilize transformer models for standard types of LLM tasks. One of them is a simple a simple translation example. The file name is genai_translation.py. Open the file either by clicking on [**genai/translation.py**](./genai/translation.py) or by entering the command below in the codespace's terminal.

```
code translation.py
```
2. Take a look at the file contents.  Notice that we are pulling in a specific model ending with 'en-fr'. This is a clue that this model is trained for English to French translation. Let's find out more about it. In a browser, go to *https://huggingface.co/models* and search for the model name 'Helsinki-NLP/opus-mt-en-fr' (or you can just go to huggingface.co/Helsinki-NLP/opus-mt-en-fr).
![model search](./images/gaidd26.png?raw=true "model search")

3. You can look around on the model card for more info about the model. Notice that it has links to an *OPUS readme* and also links to download its original weights, translation test sets, etc.

4. When done looking around, go back to the repository and look at the rest of the *translation.py* file. What we are doing is loading the model, the tokenizer, and then taking a set of random texts and running them through the tokenizer and model to do the translation. Go ahead and execute the code in the terminal via the command below.
```
python translation.py
```
![translation by model](./images/gaidd41.png?raw=true "translation by model")
 
5. There's also an example program for doing classification. The file name is classification.py. Open the file either by clicking on [**genai/classification.py**](./genai/classification.py) or by entering the command below in the codespace's terminal.

```
code classification.py
```
6. Take a look at the model for this one *joeddav/xlm-roberta-large-xnli* on huggingface.co and read about it. When done, come back to the repo.

7. This uses a HuggingFace pipeline to do the main work. Notice it also includes a list of categories as *candidate_labels* that it will use to try and classify the data. Go ahead and run it to see it in action. (This will take awhile to download the model.) After it runs, you will see each topic, followed by the ratings for each category. The scores reflect how well the model thinks the topic fits a category. The highest score reflects which category the model thinks fit best.
```
python classification.py
```
![classification by model](./images/gaidd42.png?raw=true "classification by model")

8. Finally, we have a program to do sentiment analysis. The file name is sentiment.py. Open the file either by clicking on [**genai/sentiment.py**](./genai/sentiment.py) or by entering the command below in the codespace's terminal.

```
code sentiment.py
```

9. Again, you can look at the model used by this one *distilbert-base-uncased-finetuned-sst-2-english* in Hugging Face.

10. When ready, go ahead and run this one in the similar way and observe which ones it classified as positive and which as negative and the relative scores.
```
python sentiment.py
```
![sentiment by model](./images/gaidd43.png?raw=true "sentiment by model")

11. If you're done early, feel free to change the texts, the candidate_labels in the previous model, etc. and rerun the models to see the results.

<p align="center">
**[END OF LAB]**
</p>
</br></br>

**Lab 5 - Using Ollama to run models locally**

**Purpose: In this lab, we’ll start getting familiar with Ollama, a way to run models locally.**

1. We already have a script that can download and start Ollama and fetch some models we'll need in later labs. Take a look at the commands being done in the *../scripts/startOllama.sh* file. 
```
cat ../scripts/startOllama.sh
```

2. Go ahead and run the script to get Ollama and start it running.
```
../scripts/startOllama.sh &
```
![starting ollama](./images/gaidd44.png?raw=true "starting ollama")

3. Now let's find a model to use.
Go to https://ollama.com and in the *Search models* box at the top, enter *llava*.
![searching for llava](./images/dga39.png?raw=true "searching for llava")

4. Click on the first entry to go to the specific page about this model. Scroll down and scan the various information available about this model.
![reading about llava](./images/dga40a.png?raw=true "reading about llava")

5. Switch back to a terminal in your codespace. While it's not necessary to do as a separate step, first pull the model down with ollama. (This will take a few minutes.)
```
ollama pull llava
```
6. Now you can run it with the command below.
```
ollama run llava
```
7. Now you can query the model by inputting text at the *>>>Send a message (/? for help)* prompt. Since this is a multimodal model, you can ask it about an image too. Try the following prompt that references a smiley face file in the repo.
```
What's in this image?  ../samples/smiley.jpg
```
(If you run into an error that the model can't find the image, try using the full path to the file as shown below.)
```
What's in this image? /workspaces/genai-dd/samples/smiley.jpg
```
![smiley face analysis](./images/gaidd45.png?raw=true "Smiley face analysis")

8. Now, let's try a call with the API. You can stop the current run with a Ctrl-D or switch to another terminal. Then put in the command below (or whatever simple prompt you want). 
```
curl http://localhost:11434/api/generate -d '{
  "model": "llava",
  "prompt": "What causes wind?",
  "stream": false
}'
```

9. This will take a minute or so to run. You should see a single response object returned. You can try out some other prompts/queries if you want.

![query response](./images/gaidd46.png?raw=true "Query response")

<p align="center">
**[END OF LAB]**
</p>
</br></br>

**Lab 6 - Working with Vector Databases**

**Purpose: In this lab, we’ll learn about how to use vector databases for storing supporting data and doing similarity searches.**

1. In our repository, we have a simple program built around a popular vector database called Chroma. The file name is vectordb.py. Open the file either by clicking on [**genai/vectordb.py**](./genai/vectordb.py) or by entering the command below in the codespace's terminal.

```
code vectordb.py
```

2. For purposes of not having to load a lot of data and documents, we've *seeded* some random data strings in the file that we're loosely referring to as *documents*. These can be seen in the *docdata* section of the file.
![data docs](./images/gaidd47.png?raw=true "Data docs")

3. Likewise, we've added some metadata in the way of categories for the data items. These can be seen in the categories section.
![data categories](./images/gaidd48.png?raw=true "Data categories")

4. Go ahead and run this program using the command shown below. This will take the document strings, create embeddings and vectors for them in the Chroma database section and then wait for us to enter a query.
```
python vectordb.py
```
![waiting for input](./images/gaidd49.png?raw=true "Waiting for input")

5. You can enter a query here about any topic and the vector database functionality will try to find the most similar matching data that it has. Since we've only given it a set of 10 strings to work from, the results may not be relevant or very good, but represent the best similarity match the system could find based on the query. Go ahead and enter a query. Some sample ones are shown below, but you can choose others if you want. Just remember it will only be able to choose from the data we gave it. The output will show the closest match from the doc strings and also the similarity and category.
```
Tell me about food.
Who is the most famous person?
How can I learn better?
```
![query results](./images/gaidd50.png?raw=true "Query results")

6. After you've entered and run your query, you can add another one or just type *exit* to stop.

7. Now, let's update the number of results that are returned so we can query on multiple topics. In the file *vectordb.py*, change line 70 to say *n_results=3,* instead of *n_results=1,*. Make sure to save your changes afterwards.

![changed number of results](./images/gaidd51.png?raw=true "Changed number of results")

8. Run the program again with *python vectordb.py*. Now you can try more complex queries or try multiple queries (separated by commas). 

![multiple queries](./images/gaidd52.png?raw=true "Multiple queries")
 
9. When done querying the data, if you have more time, you can try modifying or adding to the document strings in the file, then save your changes and run the program again with queries more in-line with the data you provided.

<p align="center">
**[END OF LAB]**
</p>
</br></br>

**Lab 7 - Working with RAG**

**Purpose: In this lab, we’ll build on the use of vector databases to parse a PDF and allow us to include it in context for LLM queries.**

1. In our repository, we have a simple program built for doing basic RAG processing. The file name is rag.py. Open the file either by clicking on [**genai/rag.py**](./genai/rag.py) or by entering the command below in the codespace's terminal.

```
code rag.py
```

2. This program reads in a PDF, parses it into chunks, creates embeddings for the chunks and then stores them in a vector database. It then adds the vector database as additional context for the prompt to the LLM. There is an example pdf named *data.pdf* in the *samples* directory. It contains the same random document strings that were in some of the other programs. You can look at it in the GitHub repo if interested. Open up https://github.com/skillrepos/genai-dd/blob/main/samples/data.pdf if interested.

3. You can now run the program and pass in the ../samples/data.pdf file. This will read in the pdf and tokenize it and store it in the vector database. (Note: A different PDF file can be used, but it needs to be one that is primarily just text. The PDF parsing being used here isn't sophisticated enough to handle images, etc.)
```
python rag.py ../samples/data.pdf
```
![reading in the pdf](./images/gaidd54.png?raw=true "Reading in the PDF")

4. The program will be waiting for a query. Let's ask it for a query about something only in the document. As a suggestion, you can try the one below.
```
What does the document say about art and literature topics?
```
5. The response should include only conclusions based off the information in the document.
![results from the doc](./images/gaidd56.png?raw=true "Results from the doc")
  
6. Now, let's ask it a query for some information that could come partly from the PDF. For example, try the query below. Then hit enter.
```
Give me 5 facts about the Mona Lisa
```
7. In the data.pdf file, there is one (and only one) fact about the Mona Lisa - an obscure one about no eyebrows. In the output, you will probably see this fact near the top as the tools pull the fact from the doc and the rest from the LLM's trained knowledge.

![5 facts about the Mona Lisa](./images/gaidd55.png?raw=true "5 facts about the Mona Lisa")
   
8. There is also a version of this file that includes a UI developed with *Streamlit*. The file name is ui_rag.py. Open the file either by clicking on [**genai/ui_rag.py**](./genai/ui_rag.py) or by entering the command below in the codespace's terminal.

```
code ui_rag.py
```
9. Like the previous program, this file will need a simple PDF file to be used. For convenience, there are two simple *text* PDFs in a GitHub gist (at https://gist.github.com/brentlaster/cd5d9fd57ecc2537f1269270ac2e228f) that you can download via the command below. You want to run this on your own machine, not in the codespace. This will put the files in a directory named *datafiles* wherever you run it.

```
git clone https://gist.github.com/brentlaster/cd5d9fd57ecc2537f1269270ac2e228f datafiles
```

10. Now you can run the version of the rag program with the UI using Streamlit and the command below. (Execute this in the codespace terminal. Streamlit should already be installed for you.)  After you start this, it will open up the webapp and you should see a popup in the codespace to allow you to easily access the app.
```
streamlit run ui_rag.py
```

11. After opeing up the website for the app, you can upload one of the PDF files and ask a question like you did before. The app will show evidence of *Running* in the upper right corner. Also, if you want, you can look back in the codespace's terminals and see the LLM being invoked.

![app running](./images/gaidd58.png?raw=true "App running")
![llm being accessed](./images/gaidd56.png?raw=true "LLM being accessed")

12. This will take a while to run. (You can just leave it running while we proceed with the next section.) When done, you should see a similar set of answers as you did running the non-UI version.

![app with_answer](./images/gaidd60.png?raw=true "App with answer")
<p align="center">
**[END OF LAB]**
</p>
</br></br>

**Lab 8 - Working with Agents**

**Purpose: In this lab, we’ll see how to use Agents to accomplish specialized tasks with LLMs.**

1. For this lab, we're going to use a smaller LLM to save some processing time. The LLM we'll be using is *mannix/phi3-mini-4k*.  We'll add it by pulling it with *ollama*. We'll also remove the *mistral* LLM to save some room on the system. In a terminal in the codespace, execute the two commands below. In our repository, we have a simple program built for doing basic RAG processing. The file name is rag.py. Open the file either by clicking on [**genai/rag.py**](./genai/rag.py) or by entering the commands below in the codespace's terminal.
```
ollama rm mistral
ollama pull mannix/phi3-mini-4k
```

2. In the repository, there are a couple of AI agent Python programs - [**genai/agent-shopper.py**](./genai/agent-shopper.py) and [**genai/agent-date.py**](./genai/agent-date.py). Go ahead and open the *agent-shopper.py* one and let's take a look at its structure. This agent is intended to find a good deal on a laptop that is less than $1000.
```
code agent-shopper.py
```

3. Looking at the code, after the imports, there is the load of the LLM, followed by some tool definitions that use a standard DuckDuckGoSearchRun function. The first named "Search Amazon" is one that limits web searches to the amazon site. The next is a more generic search function. Unfortunately, these searches can be hit or miss depending on API limiting by DuckDuckGo and other factors. But they show one way to define tools.

4. The callback function is on to show the raw output when an agent finishes a task. This is followed by definitions for the CrewAI Agent, Task and a call to the Crew kickoff method to start things running and then print the results. Go ahead and run the agent program via the command below.
```
python agent-shopper.py
```
5. This will run for a long time. If we do not run into issues with the search calls, it should eventually complete with a recommendation for a laptop. You may not see output on the screen changing for several minutes. However, the real value of running this is seeing the *Thoughts* and *Actions* that the agent decides on. Those should pop up on the screen in colored text along the way.  Also, if you look in the terminal where Ollama was started most recently, you'll be able to see the LLM being started and rest calls happening.

6. You can leave this running and proceed to the next steps in a separate terminal by clicking on the "+" sign over in the upper right corner of the terminal section. In a new terminal, open up our implementation of a multi-agent crew by opening [**genai/crew.py**](./genai/crew.py).
```
code crew.py
```

7. This program should look similar to the singe agent one, except it defines multiple agents, and the invocation for the task calls a separate function to do the work. Go ahead and run it in the new terminal.
```
python crew.py
```

8. Again, this will take many minutes to run. But the interesting parts will be the *Thoughts* and *Actions* that are generated as it is running. Note that the agents are deciding for themselves what to do, how to change the prompt or query, etc.

9. You can leave these running if you want and they should eventually complete, thought it may take a long time. Or you can cancel them. If you want, you can also try out the *agent-date.py* agent implementation. This one tries to determine what day of the month Thanksgiving will be on in the current year. It is not recommended to have all 3 of the agent/crew programs running at the same time.

10. When you are done, if you want, you can try tweaking some values or settings in the various programs.

<p align="center">
**[END OF LAB]**
</p>
</br></br>   

<p align="center">
**THANKS!**
</p>
