Lan: `py` From`dl/RAG/PIKE-RAG/src\pikerag\prompts\ircot\ircot.py`

Okay, let's break down this code. This code defines a communication protocol for an AI assistant that answers questions using a Chain-of-Thought (CoT) approach, specifically tailored for information retrieval. The core idea is to guide the AI to reason step-by-step, referencing relevant context to arrive at the final answer.

**1. `MessageTemplate` and `ircot_template`:**

```python
from pikerag.prompts import BaseContentParser, CommunicationProtocol, MessageTemplate

ircot_template = MessageTemplate(
    template=[
        ("system", "You are a helpful AI assistant good at question-answering."),
        ("user", """
# Task
Your task is to output either a continuous reasoning sentence or the final answer to the given question. Four demonstrations would be provided, followed by the question to be answered and the reference context you can refer to.

# Demonstration
Question: When was Neville A. Stanton's employer founded?
Reference context:
  1. Title: The Last Horse. The Last Horse (Spanish:El último caballo) is a 1950 Spanish comedy film directed by Edgar Neville starring Fernando Fernán Gómez.
  2. Title: Southampton. The University of Southampton, which was founded in 1862 and received its Royal Charter as a university in 1952, has over 22,000 students. The university is ranked in the top 100 research universities in the world in the Academic Ranking of World Universities 2010. In 2010, the THES - QS World University Rankings positioned the University of Southampton in the top 80 universities in the world. The university considers itself one of the top 5 research universities in the UK. The university has a global reputation for research into engineering sciences, oceanography, chemistry, cancer sciences, sound and vibration research, computer science and electronics, optoelectronics and textile conservation at the Textile Conservation Centre (which is due to close in October 2009.) It is also home to the National Oceanography Centre, Southampton (NOCS), the focus of Natural Environment Research Council-funded marine research.
  3. Title: Stanton Township, Champaign County, Illinois. Stanton Township is a township in Champaign County, Illinois, USA. As of the 2010 census, its population was 505 and it contained 202 housing units.
  4. Title: Neville A. Stanton. Neville A. Stanton is a British Professor of Human Factors and Ergonomics at the University of Southampton. Prof Stanton is a Chartered Engineer (C.Eng), Chartered Psychologist (C.Psychol) and Chartered Ergonomist (C.ErgHF). He has written and edited over a forty books and over three hundered peer-reviewed journal papers on applications of the subject. Stanton is a Fellow of the British Psychological Society, a Fellow of The Institute of Ergonomics and Human Factors and a member of the Institution of Engineering and Technology. He has been published in academic journals including "Nature". He has also helped organisations design new human-machine interfaces, such as the Adaptive Cruise Control system for Jaguar Cars.
  5. Title: Finding Nemo. Finding Nemo Theatrical release poster Directed by Andrew Stanton Produced by Graham Walters Screenplay by Andrew Stanton Bob Peterson David Reynolds Story by Andrew Stanton Starring Albert Brooks Ellen DeGeneres Alexander Gould Willem Dafoe Music by Thomas Newman Cinematography Sharon Calahan Jeremy Lasky Edited by David Ian Salter Production company Walt Disney Pictures Pixar Animation Studios Distributed by Buena Vista Pictures Distribution Release date May 30, 2003 (2003 - 05 - 30) Running time 100 minutes Country United States Language English Budget $94 million Box office $940.3 million
Rationale: The employer of Neville A. Stanton is University of Southampton. The University of Southampton was founded in 1862.
Answer: 1862

Question: What is the headquarters for the organization who sets the standards for ISO 21500?
Reference Context:
  1. Title: ISO 21500. ISO 21500:2012, Guidance on Project Management, is an international standard developed by the International Organization for Standardization, or ISO starting in 2007 and released in 2012. It was intended to provide generic guidance, explain core principles and what constitutes good practice in project management. The ISO technical committee dealing with project management, ISO/PC 236 was held by the American National Standards Institute (ANSI) which had approved four standards that used PMI materials. one of which was ANSI/PMI 99-001-2008, A Guide to the Project Management Body of Knowledge - 4th Edition (PMI BoK® Guide - 4th Edition) (revision and re-designation of ANSI/PMI 99-001-2004): 11/20/2008.
  2. Title: ISO 3166-2:GH. ISO 3166-2:GH is the entry for Ghana in ISO 3166-2, part of the ISO 3166 standard published by the International Organization for Standardization (ISO), which defines codes for the names of the principal subdivisions (e.g., provinces or states) of all countries coded in ISO 3166-1.
  3. Title: ISO 4031. ISO 4031 is an international standard first issued in 1978 by the International Organization for Standardization. It defined the representation of local time differentials, commonly referred to as time zones. It has since been superseded by a newer standard, ISO 8601. This newer standard sets out the current formats for local time differentials and so ISO 4031 is no longer in use.
  4. Title: ISO/TC 68. ISO/TC 68 is a technical committee formed within the International Organization for Standardization (ISO), of Geneva, Switzerland, tasked with developing and maintaining international standards covering the areas of banking, securities, and other financial services. As the standards organization under ISO responsible for the development of all international financial services standards, ISO/TC 68 plays a key role in the development and adoption of new technologies in the banking, brokerage and insurance industries. Many of its current work projects involve developing ecommerce standards such as better online security for financial transactions, XML standards for financial transactions and standards to reduce the cost and delays of international financial transactions. The membership of ISO/TC 68, consists of more than 30 organizations assigned by participating national standards bodies plus additional international standards development organizations that work collaboratively toward global financial services standards development.
  5. Title: ISO 3166-1. ISO 3166-1 is part of the ISO 3166 standard published by the International Organization for Standardization (ISO), and defines codes for the names of countries, dependent territories, and special areas of geographical interest. The official name of the standard is "Codes for the representation of names of countries and their subdivisions – Part 1: Country codes". It defines three sets of country codes:
Rationale: The standards for ISO 21500 were set by International Organization for Standardization. The International Organization for Standardization has headquarters in Geneva.
Answer: Geneva

Question: In which county was the birthplace of the Smoke in tha City performer?
Reference Context:
  1. Title: Cherokee City, Arkansas. Cherokee City is an unincorporated census-designated place in Benton County, Arkansas, United States. As of the 2010 census, its population is 72. It is the location of (or is the nearest community to) Coon Creek Bridge, which is located on Cty Rd. 24 and is listed on the National Register of Historic Places. The community was named for the Cherokee Indians, since the Trail of Tears crossed the landscape when the Cherokee migrated west to Indian territory, now Oklahoma in the late 1830s. The town is about 5 miles east of Oklahoma and 4 miles south of the Missouri state line.
  2. Title: Compton, California. Compton is a city in southern Los Angeles County, California, United States, situated south of downtown Los Angeles. Compton is one of the oldest cities in the county and on May 11, 1888, was the eighth city to incorporate. As of the 2010 United States Census, the city had a total population of 96,456. It is known as the "Hub City" due to its geographic centrality in Los Angeles County. Neighborhoods in Compton include Sunny Cove, Leland, Downtown Compton, and Richland Farms. The city is generally a working class city with some middle-class neighborhoods, and is home to a relatively young population, at an average 25 years of age, compared to the American median age of 38 (based on 2018 data).
  3. Title: MC Eiht. Aaron Tyler (born May 22, 1971), better known by his stage name MC Eiht, is an American rapper and actor. Many of his songs are based on his life in Compton. His stage name was partly inspired by the numeral in KRS-One's name. He chose Eiht for its links to "hood culture", including Olde English 800 (8 Ball) and .38 caliber firearms. He is the "de facto" leader of West Coast hip hop group Compton's Most Wanted, which also included fellow Compton-based rappers Boom Bam, Tha Chill, DJ Mike T, DJ Slip and Ant Capone. He is also known for his role as A-Wax in the 1993 film "Menace II Society".
  4. Title: Smoke in tha City. Smoke in tha City is the ninth studio album by American rapper MC Eiht, released September 14, 2004 on Factor House Records. It was produced by Black C-Zer and Quincy Miller. The album featuring guest performances by West Coast heavy-weights: RBX, Spice 1, Kokane, Jayo Felony and Daz Dillinger.
  5. Title: Beyoncé. On January 7, 2012, Beyoncé gave birth to her first child, a daughter, Blue Ivy Carter, at Lenox Hill Hospital in New York. Five months later, she performed for four nights at Revel Atlantic City's Ovation Hall to celebrate the resort's opening, her first performances since giving birth to Blue Ivy.
  6. Title: Olsztyn Voivodeship. Olsztyn Voivodeship () was an administrative division and unit of local government in Poland in the years 1945-75, and a new territorial division between 1975–1998, superseded by Warmian-Masurian Voivodeship. Its capital city was Olsztyn.
Rationale: The performer of Smoke in tha City is MC Eiht. MC Eiht's birthplace is Compton. Compton is located in the county of Los Angeles County.
Answer: Los Angeles County

Question: What weekly publication in the Connecticut city with the most Zagat rated restaurants is issued by university of America-Lite: How Imperial Academia Dismantled Our Culture's author?
Reference Context:
  1. Title: New Haven, Connecticut. New Haven is served by the daily New Haven Register, the weekly "alternative" New Haven Advocate (which is run by Tribune, the corporation owning the Hartford Courant), the online daily New Haven Independent, and the monthly Grand News Community Newspaper. Downtown New Haven is covered by an in-depth civic news forum, Design New Haven. The Register also backs PLAY magazine, a weekly entertainment publication. The city is also served by several student-run papers, including the Yale Daily News, the weekly Yale Herald and a humor tabloid, Rumpus Magazine. WTNH Channel 8, the ABC affiliate for Connecticut, WCTX Channel 59, the MyNetworkTV affiliate for the state, and Connecticut Public Television station WEDY channel 65, a PBS affiliate, broadcast from New Haven. All New York City news and sports team stations broadcast to New Haven County.
  2. Title: Imperial College London. Imperial College Union, the students' union at Imperial College, is run by five full-time sabbatical officers elected from the student body for a tenure of one year, and a number of permanent members of staff. The Union is given a large subvention by the university, much of which is spent on maintaining around 300 clubs, projects and societies. Examples of notable student groups and projects are Project Nepal which sends Imperial College students to work on educational development programmes in rural Nepal and the El Salvador Project, a construction based project in Central America. The Union also hosts sports-related clubs such as Imperial College Boat Club and Imperial College Gliding Club.
  3. Title: The End of Education. The End of Education is a book by Neil Postman about public education in America. The use of the word "end" in the title has two meanings: primarily, as a synonym for "purpose", but also as a prediction about the future of public schools if they do not successfully identify and communicate a convincing purpose for their existence within our culture.
  4. Title: America-Lite. America-Lite: How Imperial Academia Dismantled Our Culture (and Ushered in the Obamacrats) is a 2012 book by David Gelernter, published by Encounter Books.
  5. Title: David Gelernter. David Hillel Gelernter (born March 5, 1955) is an American artist, writer, and professor of computer science at Yale University. He is a former national fellow at the American Enterprise Institute and senior fellow in Jewish thought at the Shalem Center, and sat on the National Endowment for the Arts. He publishes widely; his work has appeared in "The Wall Street Journal", "New York Post", "Los Angeles Times", "The Weekly Standard", "Frankfurter Allgemeine Zeitung", and elsewhere. His paintings have been exhibited in New Haven and Manhattan.
  6. Title: Ann Arbor, Michigan. Current publications in the city include the Ann Arbor Journal (A2 Journal), a weekly community newspaper; the Ann Arbor Observer, a free monthly local magazine; the Ann Arbor Independent, a locally owned, independent weekly; and Current, a free entertainment-focused alt-weekly. The Ann Arbor Business Review covers local business in the area. Car and Driver magazine and Automobile Magazine are also based in Ann Arbor. The University of Michigan is served by many student publications, including the independent Michigan Daily student newspaper, which reports on local, state, and regional issues in addition to campus news.
  7. Title: New Haven, Connecticut. Livability.com named New Haven as the Best Foodie City in the country in 2014. There are 56 Zagat-rated restaurants in New Haven, the most in Connecticut and the third most in New England (after Boston and Cambridge). More than 120 restaurants are located within two blocks of the New Haven Green. The city is home to an eclectic mix of ethnic restaurants and small markets specializing in various foreign foods. Represented cuisines include Malaysian, Ethiopian, Spanish, Belgian, French, Greek, Latin American, Mexican, Italian, Thai, Chinese, Japanese, Vietnamese, Korean, Indian, Jamaican, Cuban, Peruvian, Syrian/Lebanese, and Turkish.
Rationale: The author of America-Lite: How Imperial Academia Dismantled Our Culture is David Gelernter. David Gelernter was educated at the Yale University. The city in Connecticut that has the highest number of Zagat-rated restaurants is New Haven. The weekly publication in New Haven that is issued by Yale University is Yale Herald.
Answer: Yale Herald

# Your Question to be answered
{content}

# Reference Context for Your Question
{context}

# Rationale Already Have
{rationale}

# Output Format
Your output should strictly follow the format below. Make sure your output parsable by json in Python.
{{
    "next_rationale": <The next sentence following to the rationale already have, ONLY one sentence.>,
    "answer": null
}}
or
{{
    "next_rationale": null,
    "answer": <Your answer to the given question>
}}

# Your Output
{limit}
""".strip())
    ],
    input_variables=["content", "context", "rationale", "limit"],
)
```

   - **`MessageTemplate`:** This class (likely from the `pikerag.prompts` module) is used to define the structure of the messages exchanged between the user and the AI assistant.  It provides a standardized way to format the prompts that are sent to the language model.
   - **`ircot_template`:** This is an instance of `MessageTemplate` that defines the specific template used for the IRCoT (Information Retrieval Chain-of-Thought) protocol.
   - **`template`:**  A list of tuples. Each tuple represents a message turn in the conversation. The first element of the tuple is the speaker ("system" or "user"), and the second element is the message content.
     - **`("system", ...)`:** The system message provides instructions and context to the AI assistant.  It sets the stage for the interaction.
     - **`("user", ...)`:** The user message contains the actual task instructions, including demonstrations (examples of question-answering with rationale), the question to be answered, and the reference context.
   - **`input_variables`:** A list of strings that specify the variables that will be dynamically inserted into the template at runtime. These variables include the question (`content`), the reference context (`context`), the existing rationale (`rationale`), and a flag to signal when a direct answer is allowed (`limit`).

**2. `IRCoTParser`:**

```python
from pikerag.utils.json_parser import parse_json

class IRCoTParser(BaseContentParser):
    def encode(
        self, content: str, rationales: List[str], references: List[str]=[], is_limit: bool=False, **kwargs,
    ) -> Tuple[str, Dict]:
        reference_strs = [f"  {i + 1}. {reference}" for i, reference in enumerate(references)]
        reference_str = "\n".join(reference_strs)
        return content, {
            "context": reference_str,
            "rationale": " ".join(rationales),
            "limit": "Null rationale allowed, output answer directly." if is_limit else "",
        }

    def decode(self, content: str, **kwargs) -> Dict[str, str]:
        try:
            output = parse_json(content)
        except Exception as e:
            print(f"[IRCoTParser] Content: {content}\nException: {e}")
            return {
                "next_rationale": None,
                "answer": None,
            }

        for key, value in output.items():
            if value is not None:
                output[key] = str(value)
        return output
```

   - **`BaseContentParser`:**  This is an abstract base class (likely from `pikerag.prompts`) that defines the interface for parsing the input and output of the language model.
   - **`IRCoTParser`:**  This class implements the `BaseContentParser` interface, providing specific logic for encoding the input and decoding the output for the IRCoT protocol.
   - **`encode`:**  This method takes the question (`content`), the list of rationales (`rationales`), and the reference contexts (`references`) as input and formats them into a dictionary that can be used to populate the `ircot_template`.
     - It formats the reference contexts into a numbered list.
     - It joins the rationales into a single string.
     - It sets the `limit` variable to indicate whether the AI should output an answer directly (without generating a rationale).
   - **`decode`:**  This method takes the raw text output from the language model (`content`) as input and attempts to parse it as a JSON object.
     - It uses the `parse_json` function (from `pikerag.utils.json_parser`) to parse the JSON.
     - If the parsing is successful, it returns a dictionary containing the extracted `next_rationale` and `answer`.
     - If parsing fails, it logs an error message and returns a dictionary with `None` values for both `next_rationale` and `answer`.
     - It converts all non-None values to strings to ensure consistency.

**3. `CommunicationProtocol` and `ircot_qa_protocol`:**

```python
ircot_qa_protocol = CommunicationProtocol(
    template=ircot_template,
    parser=IRCoTParser(),
)
```

   - **`CommunicationProtocol`:**  This class (likely from `pikerag.prompts`) combines a `MessageTemplate` and a `BaseContentParser` to define a complete communication protocol for interacting with the language model.
   - **`ircot_qa_protocol`:**  This is an instance of `CommunicationProtocol` that uses the `ircot_template` and `IRCoTParser` to define the specific protocol for the IRCoT question-answering task.

**In Summary:**

This code sets up a structured way to interact with a language model for question-answering using an Information Retrieval Chain-of-Thought approach.

-   The `MessageTemplate` defines the format of the prompts sent to the language model, including the system instructions, the question, the context, and the expected output format.
-   The `IRCoTParser` handles the encoding of the input and the decoding of the output, ensuring that the data is properly formatted and parsed.
-   The `CommunicationProtocol` combines the template and the parser to create a complete communication protocol that can be used to interact with the language model.

This architecture allows for a clear separation of concerns and makes it easier to manage the communication with the language model. The use of demonstrations helps the language model understand the desired behavior and the expected output format. The JSON-based output format ensures that the results can be easily parsed and processed.

**How it is used (Conceptual Demo):**

Imagine you have a question and a set of documents (the reference context). Here's how you might use this code:

1.  **Prepare the Inputs:**

```python
question = "What year did Finding Nemo come out?"
references = [
    "Title: The Last Horse. ...",
    "Title: Finding Nemo. ... Release date May 30, 2003 (2003 - 05 - 30) ...",
    "Title: Stanton Township, Champaign County, Illinois. ...",
    "Title: Neville A. Stanton. ...",
    "Title: Southampton. ..."
]
rationale = []  # Start with an empty rationale

```

2.  **Encode the Input:**

```python
encoded_input = ircot_qa_protocol.parser.encode(question, rationale, references)
prompt_content, prompt_data = encoded_input
```

3.  **Format the Prompt:**

```python
full_prompt = ircot_qa_protocol.template.format(content=prompt_content, **prompt_data)
```

4.  **Send to the Language Model:**

```python
# Assuming you have a function to interact with the language model
lm_output = your_language_model(full_prompt) # This is a placeholder

```

5.  **Decode the Output:**

```python
decoded_output = ircot_qa_protocol.parser.decode(lm_output)
```

6.  **Process the Results:**

```python
if decoded_output["next_rationale"]:
    rationale.append(decoded_output["next_rationale"])
    # Go back to step 2 to generate the next rationale sentence
    # (This is the iterative Chain-of-Thought process)
elif decoded_output["answer"]:
    final_answer = decoded_output["answer"]
    print(f"Final Answer: {final_answer}")
else:
    print("Could not generate rationale or answer.")

```

The key is the iterative nature. If `next_rationale` is returned, you append it to your list of rationales and call the language model again, feeding in the current question, context, and *accumulated* rationale. This continues until the model outputs an `answer` directly.
