'''
ok this is very out there so stay with me heree...
The general idea is that there are a lot of different regions in the 2 or even 3d memory space and we want to be able to tell what is related to what through proximity. For example, 
    the first memory hozie ever gets is a visual one looking at me and I introduce myself we chat about basic things like who I am and who he is. 
This memory gets placed in the very center of the space. (0,0,0) with a radius of 1.
Then, as we continue to talk, the other memories spread out from here like tree roots. BUT there can be overlap too. So if I talk about my favorite color, that memory will be placed near the middle and the first memory might get bigger as more things build on it. So after telling it my favorite color, there is a new memory at (1,0,0) with a small radius since is isn't very important so a raduus of 0.1. Then, the original first memory will be a little bigger say 1.1. Now there is overlap between these. This grows until we eventually have a structure like the one below. Now finding these regions should be easy if we start with the biggest regions (become core memories) The core memories are always fed in as context to the llm and then the smaller ones are queried when there is a specific reference to that memory. 

- a a a - - - - - -
- a b a - - y z y - 
- - - - - - - y y -
- - - - - - - - - -
- - - x x x - - - -
- - - x x o - - - -
- - - x x x - - - -
- e e e e e - - - -
- e e e f f - - - -
- g e e f f - - - -
- e e e e e - - - -

The thing with this is it needs to be fluid. The mind is always changing and we need to account for that by being able to add new memories and remove old one and even modify ones if they are referenced often. You will remember something better if people are constantly talking about it. This will make those memories grow and become more important. Conversely, if you never talk about something, it will shrink, but never removed completely, smoe details might be removed but the memory will be there (it might fade as it does in humans).

Emotional Weight: Some memories will have a higher growth rate than others based on emotional weight. For example, a traumatic event might grow rapidly in size and importance and shrink very slowly, while a mundane event might shrink over time.

Color: Some memories will have different colors to represent their nature or significance. For example, a happy memory might be bright yellow, while a sad memory might be a darker blue. This color coding can help in quickly identifying the emotional context of a memory.

State based memory: Humans remember things with associations so if I am studying late at night, I will remember it better late at night than in the morning. This can be represented by associating memories with certain states or contexts, such as time of day, location, or emotional state.

Unconscious Influences (This one I will need to be careful with): Some memories could influence behavior without being directly accessible. Like if someone was rude when discussing a topic, future conversations about that topic might be slightly more cautious, even if Hozie doesn't explicitly remember why.

Dream mechanics: Maybe during a sleeping period, memories can reogranize or create new connections between each other. This could be a way to strengthen certain memories or create new associations, similar to how dreams can help process information and emotions in humans.

'''