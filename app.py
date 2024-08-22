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
    'joy' : ['happiness', 'elation', 'bliss', 'contentment', 'delight', 'cheerfulness',
    'ecstasy', 'jubilation', 'pleasure', 'exhilaration', 'enthusiasm', 'glee',
    'gratitude', 'satisfaction', 'radiance', 'joy', 'mirth', 'rapture',
    'upbeat', 'positive', 'euphoria', 'joyous', 'celebration', 'hilarious',
    'jovial', 'smile', 'laughter', 'bright', 'optimistic', 'buoyant',
    'serenity', 'gleaming', 'triumph', 'cheer', 'elated', 'content',
    'happy', 'enjoyment', 'sublime', 'festive', 'animated', 'vivacious',
    'gleeful', 'jubilant', 'overjoyed', 'pleased', 'radiant', 'rhapsody',
    'joyful', 'brightened', 'tickled', 'satisfied', 'pleasurable', 'blissful',
    'glad', 'cheery', 'effervescent', 'merry', 'bright-eyed', 'exultant',
    'delighted', 'buoyed', 'spirit', 'joyousness', 'upbeat', 'gratified',
    'heartwarming', 'ecstatic', 'happy-go-lucky', 'inspired', 'cheerful',
    'radiant', 'contented', 'brighten', 'spirit-lifting', 'positive-vibes',
    'happy-hearted', 'elation-filled', 'exhilarated', 'happy-faced', 'gleam',
    'vibrant', 'uplifted', 'joviality', 'joyful-heart', 'festivity']
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

    # Add random emotion with 20% to pie chart
    total_emotion_score = sum(emotion_counts.values())
    random_emotion = random.choice(list(emotion_keywords.keys()))
    
    # Ensure the random emotion is the only one on the chart and capped at 20%
    emotion_counts = defaultdict(int)
    emotion_counts[random_emotion] = max(int(total_emotion_score * 0.2), 1)  # Ensure at least 1% if no other emotion

    return emotion_counts, random_emotion

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']
    sentiment_score = sia.polarity_scores(text)['compound']
    
    # Analyze emotion
    emotion_counts, dominant_emotion = analyze_emotion(text)
    if dominant_emotion == 'confusion':
        sentiment_result = "Overall sentiment is negative. Confusion often leads to negative feelings."
    else:
        sentiment_result = (
            "Overall sentiment is positive. You're feeling good!" if sentiment_score > 0.5 else
            "Overall sentiment is negative. Something might be troubling you." if sentiment_score < -0.5 else
            "Sentiment is neutral. Nothing too intense in your mood."
        )

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
    if dominant_emotion == 'confusion':
        sentiment_result = "Overall sentiment is negative. Confusion often leads to negative feelings."
    else:
        sentiment_score = sia.polarity_scores(text)['compound']
        sentiment_result = (
            "Overall sentiment is positive. You're feeling good!" if sentiment_score > 0.5 else
            "Overall sentiment is negative. Something might be troubling you." if sentiment_score < -0.5 else
            "Sentiment is neutral. Nothing too intense in your mood."
        )

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
        'rage': (
            "When dealing with rage, it's crucial to find healthy outlets for your anger. Try taking deep breaths or practicing mindfulness to calm yourself. "
            "Understanding the root cause of your anger can help you address it more effectively. Consider talking with someone you trust or seeking professional guidance if needed. "
            "Engaging in physical activities like exercise can also help release pent-up frustration."
        ),
        'despair': (
            "In moments of despair, reaching out for support is vital. Surround yourself with friends and family who can offer encouragement and understanding. "
            "Professional counseling can provide a safe space to work through your feelings and develop coping strategies. "
            "Engage in activities that bring you comfort or joy, even if they seem minor. Taking small steps toward self-care can gradually improve your outlook."
        ),
        'grief': (
            "Grief is a deeply personal experience, and it's important to give yourself permission to feel and process your emotions. Surround yourself with a supportive network of loved ones. "
            "Consider joining a support group where you can share your experience with others who understand. Allow yourself time and patience as you navigate through this challenging period. "
            "Finding ways to honor and remember what you've lost can also be a meaningful part of the healing process."
        ),
        'fear': (
            "Facing fear involves understanding what triggers it and taking gradual steps to confront it. Start by breaking the fear down into smaller, manageable parts and address each one at your own pace. "
            "Practicing relaxation techniques such as deep breathing or meditation can help reduce anxiety. It's also helpful to talk to someone you trust or seek professional help if fear is overwhelming."
        ),
        'disgust': (
            "When feeling disgust, it's often beneficial to identify the source and address it directly if possible. If you can, remove yourself from the situation causing the reaction. "
            "Engage in activities that shift your focus to positive or enjoyable experiences. Talking with someone who can provide a different perspective might also help you process these feelings more effectively."
        ),
        'sadness': (
            "Dealing with sadness involves acknowledging your emotions and seeking support from those around you. Engaging in self-care practices and activities that bring you comfort can also be helpful. "
            "It’s important to give yourself permission to rest and heal at your own pace. If sadness persists or feels overwhelming, consider reaching out to a mental health professional for additional support."
        ),
        'guilt': (
            "Handling guilt starts with reflecting on the situation and understanding what led to your feelings. If possible, make amends or take actions to rectify the situation. "
            "Forgiving yourself is crucial; everyone makes mistakes, and learning from them is part of personal growth. Talking with a trusted friend or counselor can also help you process and move past guilt."
        ),
        'confusion': (
            "Confusion often arises from complex situations or a lack of clarity. Take some time to reflect on the source of your confusion and break down the problem into smaller, more manageable parts. "
            "It can be helpful to write down your thoughts and questions to gain perspective. Seek advice from others who might offer clarity or different viewpoints. Engaging in activities that help you relax and clear your mind can also aid in resolving confusion."
        ),
        'joy': (
            "When you're feeling joyful, it's a great opportunity to embrace and celebrate the positive emotions you're experiencing. Share your happiness with those around you, as spreading joy can enhance your own sense of well-being. "
            "Engage in activities that amplify your joy, whether it’s pursuing hobbies, spending time with loved ones, or enjoying nature. Reflect on the sources of your happiness and consider how you can continue to foster such positive experiences in your life."
        )
    }
    return tips.get(emotion, "No tips available for this emotion.")


if __name__ == '__main__':
    app.run(debug=True)