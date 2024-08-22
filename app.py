import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-GUI rendering

from flask import Flask, render_template, request
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import random
from collections import defaultdict
from difflib import SequenceMatcher

# Initialize Flask app and NLTK tools
app = Flask(__name__)
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Predefined list of random situations
situations = [
    "You just found out that you got a promotion at work. How do you feel?",
    "Your best friend just moved to another city. What are your thoughts?",
    "You missed your flight due to a traffic jam. What's your reaction?",
    "You received an unexpected gift from a close friend. What do you say?",
    "Someone cut in line in front of you at the grocery store. How do you respond?",
    "You found a wallet on the street. What do you do?",
    "You are about to give a big presentation. How do you feel?",
    "Your favorite sports team just lost a major game. What's your reaction?",
    "You just finished reading a very emotional book. How do you feel?",
    "A stranger helped you with a flat tire. What do you say?",
    "You witnessed an accident on your way home. How do you react?",
    "Your friend forgot your birthday. How do you feel?",
    "You won a small prize in a lottery. What's your reaction?",
    "You are stuck in a traffic jam. What goes through your mind?",
    "Your colleague received an award you were hoping for. How do you feel?",
    "You are planning a surprise party for a friend. What are you thinking?",
    "You just finished a tough workout. How do you feel?",
    "Your pet is acting very strange. How do you react?",
    "You just moved into a new house. What's on your mind?",
    "You received a job offer from your dream company. How do you feel?",
    "You are waiting for the results of an important exam. What's on your mind?",
    "Your favorite TV show just got canceled. How do you react?",
    "You just found out you are going on a surprise vacation. How do you feel?",
    "You accidentally broke something valuable. What's your reaction?",
    "You are having a heated argument with a friend. How do you feel?",
    "You just received a compliment from someone you admire. What's your reaction?",
    "You got caught in a sudden rainstorm without an umbrella. How do you feel?",
    "You are about to try something new and challenging. What's going through your mind?",
    "Your project at work got rejected. How do you feel?",
    "You found out someone was gossiping about you. What's your reaction?"

]

# Emotion keyword lists with 95 words each
emotion_keywords = {
    'rage': ['angry', 'furious', 'enraged', 'infuriated', 'irate', 'irritated', 'incensed', 'mad', 'outraged', 'wrathful', 
             'fuming', 'livid', 'vengeful', 'hateful', 'hostile', 'aggressive', 'offended', 'bitter', 'annoyed', 'resentful', 
             'irritable', 'frustrated', 'provoked', 'exasperated', 'stormy', 'boiling', 'tempestuous', 'ballistic', 'seething', 
             'explosive', 'raging', 'cross', 'rancorous', 'irascible', 'belligerent', 'bellicose', 'antagonistic', 'truculent', 
             'pissed', 'ranting', 'raving', 'storming', 'berating', 'damning', 'condemning', 'denouncing', 'fierce', 'ferocious', 
             'violent', 'destructive', 'murderous', 'bloodthirsty', 'savage', 'barbaric', 'warring', 'hostile', 'spiteful', 
             'brutal', 'cruel', 'cold-blooded', 'mean', 'merciless', 'vicious', 'relentless', 'unforgiving', 'unyielding', 
             'implacable', 'vindictive', 'grudge', 'revenge', 'reprisal', 'retaliation', 'payback', 'wrath', 'rage', 'raging', 
             'angry', 'hostility', 'belligerence', 'combativeness', 'fight', 'battle', 'attack', 'onslaught', 'offensive', 
             'defensive', 'strike', 'hit', 'punch', 'kick', 'assault', 'battery', 'brawl', 'duel', 'conflict', 'dispute', 
             'feud', 'quarrel', 'row', 'tiff', 'wrangle', 'run-in', 'clash', 'confrontation', 'affray', 'melee', 'scuffle', 
             'altercation', 'ruckus', 'fracas', 'uproar', 'havoc', 'mayhem', 'chaos', 'frenzy', 'pandemonium', 'turmoil', 
             'disorder', 'riot', 'insurrection'],

    'despair': ['hopeless', 'desperate', 'forlorn', 'abandoned', 'forsaken', 'neglected', 'lonely', 'isolation', 'solitude', 
                'bleak', 'gloom', 'dismal', 'dismay', 'despondent', 'dejection', 'downcast', 'downhearted', 'sad', 'melancholy', 
                'mournful', 'grief', 'sorrow', 'woe', 'heartache', 'heartbroken', 'disheartened', 'heartsick', 'heavy-hearted', 
                'depressed', 'down', 'depressive', 'self-harm', 'pain', 'ache', 'hurting', 'torment', 'anguish', 'misery', 
                'suffering', 'agony', 'excruciating', 'harrowing', 'distress', 'desolation', 'ruin', 'destroyed', 'shattered', 
                'crushed', 'wrecked', 'demolished', 'devastated', 'catastrophe', 'calamity', 'tragedy', 'ill-fated', 'dead', 
                'dying', 'suicide', 'homicide', 'self-destruction', 'nihilistic', 'void', 'emptiness', 'meaninglessness', 
                'purposeless', 'pointlessness', 'futility', 'inescapable', 'doomed', 'cursed', 'condemned', 'wretched', 
                'pathetic', 'pitiful', 'sorry', 'deplorable', 'lamentable', 'heartbreaking', 'gut-wrenching', 'tear-jerking', 
                'tearful', 'sobbing', 'crying', 'wailing', 'weeping', 'lamenting', 'bewailing', 'funeral', 'obituary', 'eulogy', 
                'loss', 'bereavement', 'grieving', 'grievance', 'mournful', 'sadness', 'sorrow', 'lament', 'regret', 'remorse', 
                'contrite', 'shame', 'humiliated', 'humiliation', 'degradation', 'degraded', 'mortified', 'mortification', 
                'debased', 'demeaned', 'abased', 'worthless', 'inferior', 'inadequate', 'insufficient', 'unworthy', 'useless', 
                'inept', 'incompetent', 'failure', 'loser', 'futility', 'vain', 'vainly', 'hopelessness', 'despair', 'desperation', 
                'despondency', 'downcast', 'downhearted', 'heartless', 'cruel', 'inhumane', 'inhumanity', 'brutal', 'barbaric', 
                'merciless', 'pitiless', 'remorseless', 'unfeeling', 'cold-hearted', 'hard-hearted', 'cold', 'callous', 
                'indifferent', 'apathetic', 'unconcerned', 'dispassionate'],

    'grief': ['sorrow', 'mourning', 'bereavement', 'sadness', 'heartbreak', 'lament', 'despair', 'anguish', 'distress', 
              'woe', 'regret', 'remorse', 'loss', 'pain', 'suffering', 'sad', 'depressed', 'dejection', 'misery', 
              'heartache', 'tearful', 'weeping', 'crying', 'lamenting', 'anguished', 'grieving', 'tears', 'cry', 
              'woeful', 'distressed', 'troubled', 'suffer', 'mournful', 'desolate', 'isolated', 'forlorn', 'dismal', 
              'gloomy', 'heartbroken', 'shattered', 'crushed', 'devastated', 'hopeless', 'abandoned', 'despondent', 
              'downcast', 'downhearted', 'disheartened', 'mourning', 'heartsick', 'unhappy', 'unfortunate', 'unlucky', 
              'dismay', 'gloom', 'depressed', 'down', 'disconsolate', 'desolate', 'bereft', 'melancholy', 'gutted', 
              'broken-hearted', 'heavy-hearted', 'regretful', 'contrite', 'sullen', 'saddened', 'pained', 'sorrowful', 
              'bewailed', 'tear-stained', 'broken', 'wretched', 'depressed', 'shattered', 'regretful', 'contrite', 
              'melancholic', 'despondency', 'disconsolate', 'low', 'hopeless', 'mourn', 'mournful', 'distressed', 
              'inconsolable', 'heart-wrenching', 'anguishing'],

    'fear': ['afraid', 'terrified', 'scared', 'frightened', 'panicked', 'apprehensive', 'anxious', 'worried', 'nervous', 
             'dreadful', 'alarmed', 'spooked', 'startled', 'shocked', 'horrified', 'apprehension', 'fearful', 'freaked-out', 
             'intimidated', 'fearsome', 'fearful', 'timid', 'trembling', 'quaking', 'shaking', 'shuddering', 'apprehensive', 
             'cowed', 'shaky', 'uneasy', 'jittery', 'spooked', 'terrified', 'dismayed', 'horrified', 'panic', 'panic-stricken', 
             'terror', 'trepidation', 'disturbed', 'distressing', 'jumpy', 'on-edge', 'timorous', 'skittish', 'fearsome', 
             'worried', 'unsettled', 'troubled', 'perturbed', 'nervous', 'alarm', 'dread', 'anxiety', 'nervousness', 'worry', 
             'tremor', 'quiver', 'shiver', 'squeamish', 'cowardly', 'shrinking', 'insecure', 'apprehensive', 'startle', 
             'fright', 'spooky', 'chilled', 'faint-hearted', 'hesitant', 'uncomfortable', 'shaky', 'creepy', 'scary', 
             'frightful', 'panicked', 'horrified', 'shudder', 'jumpy', 'dreadful', 'spooked', 'distressed'],

    'disgust': ['repulsed', 'revolted', 'nauseated', 'sickened', 'appalled', 'grossed-out', 'displeased', 'disturbed', 
                'offended', 'abhorred', 'disdainful', 'loathing', 'horrified', 'disgusted', 'sick', 'repelled', 'abhorrent', 
                'repulsive', 'sickening', 'yucky', 'foul', 'nasty', 'vile', 'detestable', 'unpleasant', 'icky', 'gross', 
                'dreadful', 'disapproving', 'shocked', 'scandalized', 'outraged', 'abhorrent', 'revolting', 'unappetizing', 
                'appalling', 'repugnance', 'distaste', 'repelling', 'disapprobation', 'distasteful', 'horrid', 'discomfiting', 
                'offensive', 'displeased', 'nauseous', 'creepy', 'nauseating', 'displeasure', 'aversion', 'disfavor', 
                'dislike', 'dismay', 'revulsion', 'depraved', 'degenerate', 'unsavory', 'unpalatable', 'unappealing', 
                'abominable', 'distasteful', 'repelling', 'repugnant', 'frightful', 'hideous', 'appalling', 'abhorred', 
                'sickening', 'repulsive', 'grotesque', 'dreadful', 'hateful', 'discomforting'],

    'sadness': ['sorrowful', 'mournful', 'despondent', 'melancholy', 'dejected', 'heartbroken', 'downcast', 'gloomy', 
                'dismal', 'blue', 'downhearted', 'depressed', 'unhappy', 'woe', 'dismay', 'distressed', 'disheartened', 
                'despair', 'regret', 'sad', 'lament', 'sullen', 'woeful', 'tearful', 'mournful', 'heavy-hearted', 'dismal', 
                'gloom', 'pained', 'down', 'disconsolate', 'forlorn', 'heartache', 'melancholic', 'distress', 'unfortunate', 
                'unlucky', 'doleful', 'morose', 'glum', 'crying', 'sobbing', 'weeping', 'desolate', 'bereaved', 'wretched', 
                'suffer', 'anguished', 'agonized', 'miserable', 'afflicted', 'tormented', 'hurt', 'sick', 'gutted', 
                'saddened', 'disheartened', 'discomforted', 'unfortunate', 'dejected', 'troubled', 'remorseful', 'bitter', 
                'heartfelt', 'contrite', 'penitent', 'unfortunate', 'regretful', 'grief', 'distressing', 'woeful', 'crushed', 
                'sadden', 'tears', 'sigh', 'mourning', 'grieving', 'downcast', 'depressed', 'anguish', 'pain', 'sorrow'],

    'guilt': ['ashamed', 'remorseful', 'contrite', 'regretful', 'guilty', 'penitent', 'abashed', 'blameworthy', 'sorry', 
              'responsible', 'shameful', 'disgraced', 'apologetic', 'degraded', 'embarrassed', 'humiliated', 'humbled', 
              'debased', 'dismayed', 'displeased', 'reproachful', 'remorse', 'compunction', 'self-reproach', 'self-condemnation', 
              'repentant', 'unworthy', 'shame', 'mortified', 'abject', 'unfortunate', 'discomforted', 'contrition', 
              'self-blame', 'self-reproach', 'deserved', 'fallible', 'condemned', 'penitence', 'fault', 'error', 
              'mistake', 'misstep', 'blunder', 'failure', 'wrongdoing', 'infraction', 'misconduct', 'transgression', 
              'wrong', 'breach', 'violation', 'delinquency', 'offense', 'crime', 'sin', 'culpable', 'blame', 'censure', 
              'disapproval', 'punishment', 'penalty', 'reprimand', 'reproach', 'discredit', 'disapproval', 'self-reproach', 
              'self-accusation', 'repentance', 'inadequacy', 'inferiority', 'ineptitude', 'incompetence', 'unfit', 'faulty', 
              'deficient', 'mishap', 'blame', 'blameworthy', 'failure', 'unworthy', 'misjudged', 'regret'],

    'confusion': ['bewildered', 'perplexed', 'puzzled', 'baffled', 'lost', 'confounded', 'uncertain', 'dazed', 
                  'disoriented', 'muddled', 'disoriented', 'flustered', 'disconcerted', 'distracted', 'mixed-up', 'foggy', 
                  'unclear', 'complicated', 'complex', 'troubled', 'disturbed', 'bewildering', 'at-sea', 'disordered', 
                  'jumbled', 'incomprehensible', 'confusing', 'knotty', 'ambiguous', 'ambivalent', 'confused', 'misunderstood', 
                  'misconstrued', 'displaced', 'lost', 'dumbfounded', 'stunned', 'bewilderment', 'flummoxed', 'nonplussed', 
                  'vexed', 'thrown-off', 'disarray', 'disrupted', 'chaotic', 'perplexity', 'confusion', 'discombobulated', 
                  'unsettled', 'troubled', 'puzzlement', 'complication', 'conundrum', 'enigma', 'dilemma', 'riddle', 
                  'quandary', 'unsure', 'hesitant', 'uncertainty', 'disorientation', 'complexity', 'vexation', 'trouble', 
                  'incoherent', 'bewilder', 'bafflement', 'mystification', 'complex', 'baffle', 'vex', 'bewildered', 'trouble', 
                  'discomfort', 'mystify', 'puzzlement', 'fuzziness', 'mystified', 'incomprehension'],
    'ecstasy': ['bliss', 'rapture', 'elation', 'joy', 'delight', 'euphoria', 'happiness', 
                'exhilaration', 'exuberance', 'jubilation', 'glee', 'blissfulness', 
                'nirvana', 'paradise', 'contentment', 'beatitude', 'felicity', 'raptness', 
                'satisfaction', 'intoxication', 'thrill', 'elevation', 'enchantment', 
                'ecstatics', 'blissout', 'ravishment', 'transport', 'blissful', 'splendor', 
                'gladness', 'mirth', 'cheerfulness', 'glory', 'wonder', 'exaltation', 
                'pleasure', 'enthusiasm', 'serenity', 'transcendence', 'blissfulness', 
                'zest', 'fervor', 'high spirits', 'spirituality', 'intense joy', 
                'profound happiness', 'divine joy', 'rapturous', 'ardor', 'blithe', 
                'divine', 'glorious', 'godlike', 'transfixing', 'shining', 'bright', 
                'splendid', 'ethereal', 'elated', 'elevated', 'enchanted', 'joyous', 
                'delighted', 'infatuation', 'engrossed', 'intense pleasure', 'rapturous joy', 
                'ecstatic', 'overjoyed', 'radiant', 'intoxicating', 'exalted', 'inspired', 
                'invigorated', 'bliss state', 'joyfulness', 'dazzling', 'resplendent', 
                'vibrant', 'full of life', 'animated', 'blissful peace', 'nirvana state', 
                'heavenly', 'sublime', 'divinity', 'perfect happiness', 'in bliss', 
                'at peace', 'elevated state', 'sublimity', 'celestial', 'otherworldly', 
                'pure joy', 'rapt']
}

# Function to determine the most present emotion based on keyword frequency
def analyze_emotion(text):
    emotion_counts = defaultdict(int)
    
    # Check for direct matches with keywords
    for emotion, keywords in emotion_keywords.items():
        for word in text.lower().split():
            for keyword in keywords:
                if SequenceMatcher(None, word, keyword).ratio() > 0.8:  # Check similarity
                    emotion_counts[emotion] += 1

    # Add random emotion with 10% to pie chart
    total_emotion_score = sum(emotion_counts.values())
    if total_emotion_score > 0:
        random_emotion = random.choice(list(emotion_keywords.keys()))
        emotion_counts[random_emotion] += max(int(total_emotion_score * 0.01), 0.1)
    
    # Determine dominant emotion
    dominant_emotion = max(emotion_counts, key=emotion_counts.get, default=None)
    return emotion_counts, dominant_emotion

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']
    sentiment_score = sia.polarity_scores(text)['compound']
    
    # Sentiment analysis result
    sentiment_result = (
        "Overall sentiment is positive. You're feeling good!" if sentiment_score > 0.4 else
        "Overall sentiment is negative. Something might be troubling you." if sentiment_score < -0.4 else
        "Sentiment is neutral. Nothing too intense in your mood."
    )

    # Analyze emotion
    emotion_counts, dominant_emotion = analyze_emotion(text)
    emotion_result = f"Dominant emotion detected: {dominant_emotion.capitalize()}" if dominant_emotion else "No dominant emotion detected."
    emotion_result += "\n\n" + get_emotion_tips(dominant_emotion) if dominant_emotion else ""

    # Generate pie chart
    fig, ax = plt.subplots()
    labels = list(emotion_counts.keys())
    sizes = list(emotion_counts.values())
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired(range(len(labels))))
    ax.axis('equal')
    
    # Save pie chart to BytesIO and encode in base64
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    return render_template('index.html', sentiment=sentiment_result, result=emotion_result, plot_url=plot_url)

@app.route('/random_situation')
def random_situation():
    situation = random.choice(situations)
    return render_template('situation.html', situation=situation)

@app.route('/analyze_situation', methods=['POST'])
def analyze_situation():
    text = request.form['response']
    
    # Analyze emotion
    emotion_counts, dominant_emotion = analyze_emotion(text)
    emotion_result = f"Dominant emotion detected: {dominant_emotion.capitalize()}" if dominant_emotion else "No dominant emotion detected."
    emotion_result += "\n\n" + get_emotion_tips(dominant_emotion) if dominant_emotion else ""

    # Generate pie chart
    fig, ax = plt.subplots()
    labels = list(emotion_counts.keys())
    sizes = list(emotion_counts.values())
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired(range(len(labels))))
    ax.axis('equal')
    
    # Save pie chart to BytesIO and encode in base64
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    return render_template('situation.html', situation=request.form['situation'], result=emotion_result, plot_url=plot_url)

def get_emotion_tips(emotion):
    tips = {
        'ecstasy': "Embrace the Moment: Savor the feelings of joy and excitement. Let yourself be fully immersed in the experience."
                   "\nShare the Joy: Spread the happiness by sharing your good news or positive energy with others."
                   "\nReflect on Your Success: Think about what led to this feeling and how you can recreate it in the future.",
        'admiration': "Express Your Gratitude: Let the person you admire know how much they inspire you. A sincere compliment can go a long way."
                      "\nLearn from Them: Identify the qualities you admire and think about how you can incorporate them into your own life."
                      "\nStay Humble: While it's great to admire others, remember to stay grounded and appreciate your own strengths too.",
        'terror': "Breathe and Ground Yourself: Focus on deep breathing to help calm your nerves. Try to ground yourself in the present moment."
                  "\nAssess the Situation: Determine if the fear is rational or if it's being amplified by your mind."
                  "\nReach Out for Support: If terror overwhelms you, seek help from friends, family, or a professional.",
        'amazement': "Appreciate the Wonder: Take a moment to soak in the beauty or significance of what amazed you."
                     "\nShare Your Experience: Talk about it with others to relive the excitement and wonder."
                     "\nKeep an Open Mind: Stay open to new experiences that can continue to amaze and inspire you.",
        'grief': "Allow Yourself to Grieve: It's okay to feel sad and mourn your loss. Give yourself permission to feel the pain."
                 "\nTalk to Someone: Sharing your feelings with friends, family, or a counselor can help ease the burden."
                 "\nRemember the Good Times: Focus on positive memories to help cope with the sadness.",
        'loathing': "Understand the Source: Reflect on why you feel such strong aversion. Is it justified or a result of misunderstanding?"
                    "\nChannel the Energy: Use that intense emotion as motivation to address the underlying issue constructively."
                    "\nPractice Empathy: Try to understand the perspective of the person or thing you loathe, it may help reduce the intensity of your feelings.",
        'rage': "Take Deep Breaths: Calm yourself with slow, deep breathing to help control the intensity of your anger."
                "\nStep Away: Remove yourself from the situation until you cool down and can think clearly."
                "\nExpress Yourself Calmly: Once calm, communicate your feelings clearly and assertively without being confrontational.",
        'confusion': "Confusion often arises from complex situations or a lack of clarity. Take some time to reflect on the source of your confusion and break down the problem into smaller, more manageable parts. It can be helpful to write down your thoughts and questions to gain perspective. Seek advice from others who might offer clarity or different viewpoints. Engaging in activities that help you relax and clear your mind can also aid in resolving confusion."
    }
    return tips.get(emotion, "No specific tips available for this emotion.")

if __name__ == '__main__':
    app.run(debug=True)
