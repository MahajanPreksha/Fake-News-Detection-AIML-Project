{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "d2c393b3",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import re\n",
    "import string\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import accuracy_score, precision_score, roc_curve, f1_score, recall_score, confusion_matrix, classification_report,  plot_confusion_matrix\n",
    "from sklearn.metrics import classification_report\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.model_selection import train_test_split, cross_val_score\n",
    "from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.model_selection import GridSearchCV"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "244ab837",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_fake = pd.read_csv(r\"C:\\Users\\hp\\Downloads\\archive\\Fake.csv\")\n",
    "df_true = pd.read_csv(r\"C:\\Users\\hp\\Downloads\\archive\\True.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "26f21ab6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>title</th>\n",
       "      <th>text</th>\n",
       "      <th>subject</th>\n",
       "      <th>date</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Donald Trump Sends Out Embarrassing New Year’...</td>\n",
       "      <td>Donald Trump just couldn t wish all Americans ...</td>\n",
       "      <td>News</td>\n",
       "      <td>December 31, 2017</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Drunk Bragging Trump Staffer Started Russian ...</td>\n",
       "      <td>House Intelligence Committee Chairman Devin Nu...</td>\n",
       "      <td>News</td>\n",
       "      <td>December 31, 2017</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Sheriff David Clarke Becomes An Internet Joke...</td>\n",
       "      <td>On Friday, it was revealed that former Milwauk...</td>\n",
       "      <td>News</td>\n",
       "      <td>December 30, 2017</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Trump Is So Obsessed He Even Has Obama’s Name...</td>\n",
       "      <td>On Christmas day, Donald Trump announced that ...</td>\n",
       "      <td>News</td>\n",
       "      <td>December 29, 2017</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Pope Francis Just Called Out Donald Trump Dur...</td>\n",
       "      <td>Pope Francis used his annual Christmas Day mes...</td>\n",
       "      <td>News</td>\n",
       "      <td>December 25, 2017</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                               title  \\\n",
       "0   Donald Trump Sends Out Embarrassing New Year’...   \n",
       "1   Drunk Bragging Trump Staffer Started Russian ...   \n",
       "2   Sheriff David Clarke Becomes An Internet Joke...   \n",
       "3   Trump Is So Obsessed He Even Has Obama’s Name...   \n",
       "4   Pope Francis Just Called Out Donald Trump Dur...   \n",
       "\n",
       "                                                text subject  \\\n",
       "0  Donald Trump just couldn t wish all Americans ...    News   \n",
       "1  House Intelligence Committee Chairman Devin Nu...    News   \n",
       "2  On Friday, it was revealed that former Milwauk...    News   \n",
       "3  On Christmas day, Donald Trump announced that ...    News   \n",
       "4  Pope Francis used his annual Christmas Day mes...    News   \n",
       "\n",
       "                date  \n",
       "0  December 31, 2017  \n",
       "1  December 31, 2017  \n",
       "2  December 30, 2017  \n",
       "3  December 29, 2017  \n",
       "4  December 25, 2017  "
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_fake.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "751c2399",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>title</th>\n",
       "      <th>text</th>\n",
       "      <th>subject</th>\n",
       "      <th>date</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>As U.S. budget fight looms, Republicans flip t...</td>\n",
       "      <td>WASHINGTON (Reuters) - The head of a conservat...</td>\n",
       "      <td>politicsNews</td>\n",
       "      <td>December 31, 2017</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>U.S. military to accept transgender recruits o...</td>\n",
       "      <td>WASHINGTON (Reuters) - Transgender people will...</td>\n",
       "      <td>politicsNews</td>\n",
       "      <td>December 29, 2017</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Senior U.S. Republican senator: 'Let Mr. Muell...</td>\n",
       "      <td>WASHINGTON (Reuters) - The special counsel inv...</td>\n",
       "      <td>politicsNews</td>\n",
       "      <td>December 31, 2017</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>FBI Russia probe helped by Australian diplomat...</td>\n",
       "      <td>WASHINGTON (Reuters) - Trump campaign adviser ...</td>\n",
       "      <td>politicsNews</td>\n",
       "      <td>December 30, 2017</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Trump wants Postal Service to charge 'much mor...</td>\n",
       "      <td>SEATTLE/WASHINGTON (Reuters) - President Donal...</td>\n",
       "      <td>politicsNews</td>\n",
       "      <td>December 29, 2017</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                               title  \\\n",
       "0  As U.S. budget fight looms, Republicans flip t...   \n",
       "1  U.S. military to accept transgender recruits o...   \n",
       "2  Senior U.S. Republican senator: 'Let Mr. Muell...   \n",
       "3  FBI Russia probe helped by Australian diplomat...   \n",
       "4  Trump wants Postal Service to charge 'much mor...   \n",
       "\n",
       "                                                text       subject  \\\n",
       "0  WASHINGTON (Reuters) - The head of a conservat...  politicsNews   \n",
       "1  WASHINGTON (Reuters) - Transgender people will...  politicsNews   \n",
       "2  WASHINGTON (Reuters) - The special counsel inv...  politicsNews   \n",
       "3  WASHINGTON (Reuters) - Trump campaign adviser ...  politicsNews   \n",
       "4  SEATTLE/WASHINGTON (Reuters) - President Donal...  politicsNews   \n",
       "\n",
       "                 date  \n",
       "0  December 31, 2017   \n",
       "1  December 29, 2017   \n",
       "2  December 31, 2017   \n",
       "3  December 30, 2017   \n",
       "4  December 29, 2017   "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_true.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "1264a712",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Inserting a column \"class\" as target feature\n",
    "df_fake[\"class\"] = 0\n",
    "df_true[\"class\"] = 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "21054d32",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((23481, 5), (21417, 5))"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Shape of dataframes:\n",
    "df_fake.shape, df_true.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "8a4656bd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(Index(['title', 'text', 'subject', 'date', 'class'], dtype='object'),\n",
       " Index(['title', 'text', 'subject', 'date', 'class'], dtype='object'))"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Column Names of dataframes:\n",
    "df_fake.columns, df_true.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "09be8af3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(title      object\n",
       " text       object\n",
       " subject    object\n",
       " date       object\n",
       " class       int64\n",
       " dtype: object,\n",
       " title      object\n",
       " text       object\n",
       " subject    object\n",
       " date       object\n",
       " class       int64\n",
       " dtype: object)"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Data types of columns:\n",
    "df_fake.dtypes, df_true.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "c0182501",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Missing Values:\n",
      "title      0\n",
      "text       0\n",
      "subject    0\n",
      "date       0\n",
      "class      0\n",
      "dtype: int64\n",
      "Missing Values:\n",
      "title      0\n",
      "text       0\n",
      "subject    0\n",
      "date       0\n",
      "class      0\n",
      "dtype: int64\n"
     ]
    }
   ],
   "source": [
    "#Missing Values Analysis\n",
    "missing_values = df_fake.isnull().sum()\n",
    "print(\"Missing Values:\")\n",
    "print(missing_values)\n",
    "\n",
    "missing_values = df_true.isnull().sum()\n",
    "print(\"Missing Values:\")\n",
    "print(missing_values)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "82bd36ee",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Unique values in title:\n",
      "MEDIA IGNORES Time That Bill Clinton FIRED His FBI Director On Day Before Vince Foster Was Found Dead                                                                  6\n",
      "ELEMENTARY SCHOOL PLANS ‘BLACKS ONLY’ FIELD TRIP TO COLLEGE FOR THIRD GRADERS                                                                                          4\n",
      "FORMER FBI ASST DIRECTOR: “Jim Comey ‘Danced With The Devil’…I’m Glad He’s Gone” [VIDEO]                                                                               4\n",
      "AN INSIDE LOOK AT OBAMA’S 5-STAR SUMMER VACATION RETREAT: Meanwhile…62% Of Americans Won’t Be Taking A Vacation This Summer                                            3\n",
      "GARY JOHNSON: Meet The “Creepy” Pro-Amnesty, Anti-Gun, Pro-TPP, Pro-Abortion, Democrat Party Operative And His Anti-Gun Rights, Friend Of Clinton’s VP Pick [VIDEO]    3\n",
      "                                                                                                                                                                      ..\n",
      " Iowa Legislator Leaves Republican Party Over Trump’s Racism                                                                                                           1\n",
      " Mark Kirk Becomes First Republican To Un-Endorse Donald Trump (VIDEO)                                                                                                 1\n",
      " Women Will Get More Elected Representation If Clinton Becomes President                                                                                               1\n",
      " Meryl Streep Impersonated Donald Trump, And It Was Absolutely PERFECT (VIDEO/TWEETS)                                                                                  1\n",
      "A Troubled King: Chicago’s Rahm Emanuel Desperate to Save His 2020 Presidential Run                                                                                    1\n",
      "Name: title, Length: 17903, dtype: int64\n",
      "\n",
      "Unique values in text:\n",
      "                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     626\n",
      "Things didn t go as well as Nicholle had hoped and now, she s lashing out at the owner of the billboard company who was inundated with calls from angry residents.Rouse & Revolt owner Nicholle Lewis told Newsweek that she didn t sleep much last night. Her phone wouldn t stop going off with the persistent buzzing of death threats for her sign, which stylizes the number 45 into a swastika and features Trump posed in a Hitler-esque stance. I m living in a small, podunk red town and I m already getting death threats,  said Lewis, whose store is in right-leaning Chico, population 90,000.  My business has completely floundered. Overnight I had more one-star reviews than all the reviews I ve received in a year. The sign was up for less than 24 hours before Stott Outdoor Advertising took it down amid a backlash that featured online attacks.But Lewis said she s standing behind her beliefs. I don t necessarily think that just because I m a business doesn t mean I can t mix my beliefs,  Lewis said.  That s a common misconception that you can t mix politics and business.   I have a platform and I m going to use it. Lewis has used the billboard at the corner of Third Avenue and Mangrove Avenue all year, though this is the first time she s made it political. He is not presidential, he is not a president,  Lewis said.  He is a celebrity who was born into money. And he s a Nazi sympathizer. I am going to stand behind my beliefs regardless. Lewis said she has been getting some support, too.Lewis got slammed by bad reviews on her store, Rouse and Revolt Facebook page, as well as her personal Facebook page, where she posted this video. In the video, Lewis pleas with liberals to make false claims against the sign company, as a payback because she s been hit with bad reviews. That s called liberal logic, in case you re not familiar with how liberal operate.Watch, as angry liberal Nicholle Haber Lewis threatens the sign company with a lawsuit while simultaneously asking libs to help her destroy his business.(function(d, s, id) {  var js, fjs = d.getElementsByTagName(s)[0];  if (d.getElementById(id)) return;  js = d.createElement(s); js.id = id;  js.src = \"//connect.facebook.net/en_US/sdk.js#xfbml=1&version=v2.10\";  fjs.parentNode.insertBefore(js, fjs);}(document, 'script', 'facebook-jssdk'));Posted by Nicholle Haber Lewis on Friday, October 6, 2017The general manager of Stott Outdoor Advertising replied to Newsweek:Jim Moravec, the general manager of Stott Outdoor Advertising, told Newsweek that the company took it down because  a lot of people misinterpreted the billboard and who the speaker was. I should have not accepted the ad in the first place,  Moravec said, adding that the sign looked more like a call for action than an ad for the clothing store.   Newsweek                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               4\n",
      "On September 15, Hillary  apparently  held a rally in the Old Student Recreational Center at the University of North Carolina in Greensboro, NC. This was her first public appearance since she convulsed and had to be lifted into her van following the 9-11 memorial in NYC. Hillary s campaign was quick to blame the heat until they discovered conservative websites were able to quickly access the weather and determine that it was a balmy 74 degrees in NYC! Hillary was quickly whisked away to her daughter Chelsea s apartment following her  incident,  passing more than one hospital on the way to  recuperate.  Did Hillary recover from her  incident  or was this rally faked to make it look like she did?Here is the first video of Hillary s Greensboro, NC rally that was causing viewers to ask  what the heck are those cameras in Hillary s  audience  pointing at?  If you look closely, it sure isn t Hillary!Watch this video first, and then the video below to get a closer analysis of what appears to be a phony rally for Hillary at https://twitter.com/WDFx2EU5/status/777263623915745280This stunning video takes the viewer through and shows step-by-step where the audience is faked:Here is the actual video from C-Span to prove nothing was doctored in the videos above. Holy moly!                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       4\n",
      "AP News  The regulars amble in before dawn and claim their usual table, the one next to an old box television playing the news on mute.Steven Whitt fires up the coffee pot and flips on the fluorescent sign in the window of the Frosty Freeze, his diner that looks and sounds and smells about the same as it did when it opened a half-century ago. Coffee is 50 cents a cup, refills 25 cents. The pot sits on the counter, and payment is based on the honor system.People like it that way, he thinks. It reminds them of a time before the world seemed to stray away from them, when coal was king and the values of the nation seemed the same as the values here, in God s Country, in this small county isolated in the foothills of the Appalachian Mountains.Everyone in town comes to his diner for nostalgia and homestyle cooking. And, recently, news reporters come from all over the world to puzzle over politics   because Elliott County, a blue-collar union stronghold, voted for the Democrat in each and every presidential election for its 147-year existence.Until Donald Trump came along and promised to wind back the clock. He was the hope we were all waiting on, the guy riding up on the white horse. There was a new energy about everybody here,  says Whitt. I still see it. Despite the president s dismal approval ratings and lethargic legislative achievements, he remains profoundly popular here in these mountains, a region so badly battered by the collapse of the coal industry it became the symbolic heart of Trump s white working-class base.The frenetic churn of the national news, the ceaseless Twitter taunts, the daily declarations of outrage scroll soundlessly across the bottom of the diner s television screen, rarely registering. When they do, Trump doesn t shoulder the blame   because the allegiance of those here is as emotional as it is economic.It means God, guns, patriotism, saying  Merry Christmas  and not Happy Holidays. It means validation of their indignation about a changing nation: gay marriage and immigration and factories moving overseas. It means tearing down the political system that neglected them again and again in favor of the big cities that feel a world away.On those counts, they believe Trump has delivered, even if his promised blue-collar renaissance has not yet materialized. He s punching at all the people who let them down for so long   the presidential embodiment of their own discontent. He s already done enough to get my vote again, without a doubt, no question,  Wes Lewis, a retired pipefitter and one of Whitt s regulars, declares as he deals the day s first hand of cards.He thinks the mines and the factories will soon roar back to life, and if they don t, he believes they would have if Democrats and Republicans and the media   all  crooked as a barrel of fishhooks    had gotten out of the way. What Lewis has now that he didn t have before Trump is a belief that his president is pulling for people like him. One thing I hear in here a lot is that nobody s gonna push him into a corner,  says Whitt, 35.  He s a fighter. I think they like the bluntness of it. He plops down at an empty table next to the card game, drops a stack of mail onto his lap and begins flipping through the envelopes. Bill, bill, bill,  he reports to his wife, Chesla, who has arrived to relieve him at the restaurant they run together. He needs to run home and change of out his Frosty Freeze uniform, the first of several work ensembles he wears each day, and put on his second, a suit and tie. He also owns a local funeral home and he s the county coroner, elected as a Democrat.The Whitts, like many people here, cobble together a living with a couple jobs each   sometimes working 12 or 15 hours a day   because there aren t many options better than minimum wage. There s the school system, and a prison, and that s pretty much it. Outside of town, population 622, roads wind past rolling farms that used to grow tobacco before that industry crumbled too, then up into the hills of Appalachia, with its spectacular natural beauty and grinding poverty that has come to define this region in the American imagination.Whitt slides a medical bill across the table. Looks like this one is the new helmet,  he says, and his wife tears the envelope open and reports the debt: $3,995. They will add it to a growing pile that s already surpassed $40,000 since their son was born nine months ago with a rare condition. His skull was shaped like an egg, the bones fused together in places they shouldn t be. Tommy, their baby boy with big blue eyes, has now outgrown three of the helmets he s been required to wear after surgery so his bones grow back together like they should.They pay $800 a month for insurance. But when they took their baby to a surgeon in Cincinnati, they learned it was out of network. In-network hospitals offered only more invasive surgeries, so they opted to pay out of pocket. At the hospital they were told that if they d been on an insurance program for the poor, it would have all been free.This represents the cracks in America s institutions that drove Whitt, a lifelong Democrat, from supporting President Barack Obama to buying a  Make America Great Again  cap that he still keeps on top of the hutch. Many of their welfare-dependent neighbors, he believes, stay trapped in a cycle of handouts and poverty while hardworking taxpayers like him and his wife are stuck with the tab and can t get ahead. Where s the fairness in that?  he asks.But Whitt doesn t blame Trump for the failure this year to repeal the health care law and replace it with something better. He blames the  brick wall  in Washington, the politicians he sees as blocking everything Trump proposes while  small people  like them in small places like this are left again to languish.A third of people here live in poverty. Just 9 percent of adults have a college degree, but they always made up for that with backbreaking labor that workers traveled dozens of miles to neighboring counties or states to do, and those jobs have gotten harder to find.Many here blame global trade agreements and the  war on coal    environmental regulations designed by Obama s administration to curb carbon emissions   for the decline of mining and manufacturing jobs. When Trump bemoans the  American carnage  of lost factories and lost faith, it feels like he s talking to the people in these Appalachian hills. When he scraps dozens of regulations to the horror of environmentalists and says it means jobs are on the way, they embrace him.Coal has ticked up since Trump took office; mining companies have added 1,200 jobs across the country since his inauguration, more than 180 of them in Kentucky. But industry analysts say that was tied largely to market forces and dismiss Trump s repeated pledges to resuscitate the coal industry as pie in the sky. Coal has been on the decline for many decades for many reasons outside of regulation: far cheaper natural gas, mechanization, thinning Appalachian seams. With the opposition he s had, I think he s pulling the plow pretty good,  offers Wes Lewis from the card table. A few months ago, he says, he saw four brand-new coal rigs going through town.  For the longest time, under Obama, all we saw were trucks being pulled on wreckers, because people turned belly up, they went broke. Lewis says he s heard about friends of friends being called back to work. He s noticed new trucks in people s driveways, too, which he takes as evidence that his neighbors are feeling confident about their futures. These tiny signs stack up to him as proof. Lewis fishes the tag out of the bib of his overalls:  Made in Mexico,  it reads. Trump s bringing them back,  he says.Lewis, a registered Democrat, trusts Trump because he trusts his values. And because of that, he trusts Trump s other promises   so strongly he can t think of anything that would shake that faith in him. If the factories and mines don t come back, he ll blame the opposition. If there isn t a wall on the Mexico border, he says, it won t be because Trump didn t try. If investigators find his campaign colluded with Russians, it s because so many people are so determined to bring him down.Go HERE to read entire story.      4\n",
      "                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       4\n",
      "                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    ... \n",
      "This is one of the most epic smack downs of a gun nut you ll ever see.During Real Time on Friday night, host Bill Maher was joined on his panel by guests Lawrence Wilkerson and Emily Miller to discuss the Orlando mass shooting and the role guns played. This tragedy was brought to you by guns and religion,  Maher said to kick off the discussion.Maher and conservative reporter Emily Miller agreed that only followers of Islam commit acts of terrorism, forgetting mass shootings that were perpetrated by white Christian males such as Dylann Roof, who killed nine black people at a church in Charleston. Or Timothy McVeigh, who committed the Oklahoma City Bombing in 1995. He grew up a Roman Catholic and took the Last Rites prior to his execution. And there are so many other examples that they could literally fill this article and more. And here s a list of violence against the LGBT community, some of which were committed by anti-gay Christians with guns.But it was Republican retired Army Colonel Lawrence Wilkerson who stole the show when he buried Miller and her gun nut views, much to Maher s amusement. We need some kind of control on the weapons in this country,  Wilkerson said.  We do not need large capacity magazines, semi-automatic weapons in the hands of anybody in this country, other than possibly law enforcement. Wilkerson admitted that he owns 14 guns at home because he inherited many them from relatives who passed away, but he only uses them for hunting and is very selective about who he sells a gun to if he wants to let one go.Miller immediately volunteered to buy some guns from Wilkerson and he quickly rejected her.  I wouldn t sell them to you,  he said.Wilkerson went on to hold Republicans responsible for the shooting to some degree because they have refused to pass a law banning people on the terrorism watch list from buying and owning guns.Miller then attempted to defend guns as a necessary tool for home defense but Wilkerson, who served in the Vietnam War, didn t buy that malarkey either, saying that he doesn t own guns to shoot people and pointing out that we have law enforcement to deal with home invasions. When Miller attempted to discuss law enforcement response time, Wilkerson shut her down. I m 71-years-old. I ve lived in this country for seventy-one years, except for the years I was deployed fighting for this country when I did need my guns, and nobody has ever entered my house and tried to kill me! After that, Miller was stunned into silence.Here s the video via YouTube.Featured image via screen capture                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    1\n",
      "Scott Baio made an appearance on Fox Business where the actor and Donald Trump supporter insinuated that President Obama is a Muslim and a terrorist sympathizer. Baio said during the interview: I can t tell whether he is dumb, a Muslim, or a Muslim sympathizer   I don t think he is dumb. Chachi s  Muslim sympathizer  comment is particularly troubling. It seems that what he meant to say is that President Obama is a  terrorist sympathizer.  I m not entirely sure what a  Muslim sympathizer  is, though I have to admit I feel particularly sympathetic towards the 1.6 billion Muslims out there who are not violent terrorists but are still demonized by xenophobic right-wingers such as Baio. I don t know what the endgame is for liberals to continue these policies. . . Is it to totally eliminate the United States as it was created and founded and as it is now? It s hard to tell exactly what policies Baio is talking about. He mentions that Hillary Clinton wants to increase something by 500% so we can assume he is talking about immigration. Though that statistic is completely absurd. In fact, more immigrants have been deported under Obama than any other president in the history of the United States by far. These mass deportations are one of the major criticisms the left has had of the Obama administration. It is one part of Obama s legacy that very few people on the left would hope to see in his successor. However, this is a segment on Fox Business, so facts like that are not mentioned.Baio goes on to say that Democrats only get angry over guns and Republicans, yet do not get angry over terrorism. That is of course nonsense. The left is furious that the NRA has taken our nation hostage, making it possible for terrorists to purchase assault weapons in a matter of minutes, then use those weapons to massacre huge numbers of people. The left also is pretty furious that Muslims are demonized by the right and the media for acts of terror and hatred despite the fact that terrorism is routinely perpetrated by members of all religions.You can watch the segment below.Featured image from screenshot via YouTube                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            1\n",
      "In the days following the massacre at the Pulse Nightclub in Orlando, Donald Trump has tried to claim that he is a better ally to LGBT Americans than Hillary Clinton. Trump said: Hillary Clinton can never claim to be a friend of the gay community as long as she continues to support immigration policies that bring Islamic extremists into our country and who suppress women, gays and anyone else who doesn t share their views or values. But a few months ago, Trump posed for a photo alongside Robert Jeffress, a right-wing pastor who is notorious for being anti-LGBT. In the photo, tweeted by Trump, the two men stand next to each other with their thumbs up.Honored to pray for my friend, @realDonaldTrump, at tonight's Dallas rally. #TrumpDallas c: @DanScavino pic.twitter.com/BcgWuszPnu  Dr. Robert Jeffress (@robertjeffress) June 17, 2016Robert Jeffress, a pastor from Dallas known for his anti-LGBT sentiment, shared a photo in which he posed with Trump at the candidate s rally at Gilley s, the city s famous honky-tonk. Honored to pray for my friend, @realDonaldTrump, at tonight s Dallas rally,  Jeffress tweeted, along with a photo in which they both held their thumbs up. Trump retweeted the image on Friday.Jeffress has pushed rhetoric on gay people eerily similar to that espoused by extremist groups like ISIS.The First Baptist Church pastor in February 2015 was quoted as saying the gay rights movement  will pave the way for that future world dictator, the Antichrist, to persecute and martyr Christians without any repercussions whatsoever. Jeffress last month celebrated his state s leaders  decision to refuse to comply with President Barack Obama s directive to create more accessibility for transgender students in public schools, saying  it s time for an all-out rebellion against this absolute tyranny of the Obama administration. In reality, anti-LGBT rhetoric has been the norm within Republican politics. All the leading candidates for the 2016 Republican presidential nomination were connected to the anti-LGBT movement, most blatantly Senator Ted Cruz, but the practice spread to the entire party.While Democrats dragged their feet to become completely in favor of LGBT equality, the party is also the first to nominate a presidential candidate   President Obama   who supported same-sex marriage. By contrast, the notion of repealing the right to same-sex marriage is still popular within the Republican party.Featured image Joe Raedle/Getty Images                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            1\n",
      "A rainbow flag raised to honor the LGBT victims in the worst mass shooting in American history is in danger of being removed from the Hillsborough County center in Tampa Bay, Florida, because a Republican official said she talked to a Christian (singular) who said the flag was  unbearable.  Priorities!The pride flag was raised just hours after Hillsborough s commissioners voted 5-1 to hang the flag for the remainder of the month both in acknowledgement of it being Pride Month and in honor of the victims of Orlando, many of whom were LGBT and whose killer was reportedly vehemently anti-gay.Rather than see this as a symbol of unity during the wake of a national tragedy, one Republican commissioner, Stacy White, decided to make this all about fake Christian oppression. Apparently, just seeing the rainbow colors has caused one of her constituents to feel like she can t even drive to work. In an email sent to her fellow commissioners, White complains about how the rainbow flag might be harming those with  strong Christian values. Warning: Her letter is bonkers.My office recently received an anonymous phone call from a county employee stating that, because of her strong Christian beliefs, it will be nearly unbearable for her pass the  pride  flag each morning as she enters the workplace. She clearly indicated that the display of that flag, for her, has created a hostile work environment.My question is, given the nature of this employee complaint, has the board s action taken this past Wednesday created any issues for the county from a Human Resources perspective? If the display of this flag exposes the county any liability whatsoever, I request that it be taken down on the basis that the board s action has violated the workplace rights of some of our employees.White closed her letter out by referring to the Pride Flag as a  divisive, politically-charged symbol  inappropriate for  our workplace. In a country that still routinely displays a flag used in the Civil War to rally around slavery on state property, you may need a microscope and a very steady hand to create a violin small enough to play for White s pity party here. It s also particularly astounding that Republicans can somehow take a hate crime perpetuated on (mostly) LGBT Latinos and somehow make it all about them.Ironically, in the days after the shooting, Republicans shamelessly attempted to leverage the raw emotions of the LGBT community to make the, frankly insulting, pitch that the gay community should vote for Republicans because they could keep them safe from Muslims. The LGBT community, having had decades of experience in picking up on when they are being pandered to, gave the GOP a resounding middle finger and instead demanded America solve a much more dangerous problem: Guns.Apparently, having realized gay people weren t taking the bait, Republicans have reverted back to bashing them instead. Now, just a week after the shooting, Commissioner Stacy White is unflinchingly accusing a symbol of their pride of being  unbearable  to Christians. In several other Republican-dominated counties across America, officials have also refused to honor President Obama s request to lower their flags to half-mast based on technicalities. An obnoxious form of grandstanding meant to signal to their most bigoted voters that they don t really like gay people, so please don t vote them out.Thankfully in Hillsborough at least the other commissioners seem to have an ounce of common decency. Faced with White s pitiable letter, they kindly told her to shove it.Featured image via Justin Sullivan/Getty Images                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              1\n",
      "21st Century Wire says It s not that far away. Guess who wants to run for president in 2020?Just before the New Year bell rung, the embattled Chicago Mayor Rahm Israel Emanuel (D) was forced to cut short his family vacation in Cuba, and headed back to the windy city after yet another embarrassing police shooting last week which claimed two lives as police responding to a  domestic disturbance  call killed  19-year-old Quintonio LeGrier, an engineering student at Northern Illinois University, and bystander 55-year-old Betti Jones, a grandmother.Last week, 100 people, family and activists, held a vigil for the victims. The Mayor has attempted a sort of mea culpa this time, but the public is not buying it, mainly because this was the second major high-profile police shooting incident to rock city hall in the last month. TROUBLE IN HIS KEEP: The LaQuan McDonald shooting has rattled the  dynasty potential  of Chicago s Rahm Emanuel.Currently serving his second term in office, the 56-year-old mayor is going from one crisis to the next, as tensions continue to simmer after the initial controversy, the killing of 17-year-old Laquan McDonald who was shot 16 times by city police on camera   which prompted angry mobs demanding  The King  step down.There could be a political price to pay for all of this. Not only are #BlackLivesMatter on the rampage over these latest shooting events, but also LeGrier s mother, Janet Cooksey, appeared on national TV last Sunday at a news conference, wearing a T-shirt emblazoned with the face of Mayor Emanuel with the words  Rahm Failed Us. Throughout the debacle, Emmanuel has not given in to the public pressure, even though his approval rating has plummeted 20% since hostilities began. On top of this, Chicago has earned the reputation of  murder capital of America , despite the mayor s staunch liberal policy of  gun-free zones  across the city which some critics credit towards creating a criminal s paradise. The Mayor is already on shaky ground. Only last winter, Emanuel narrowly survived a close run-off election to keep his job in office.All of this has the potential to cast a shadow over Emanuel s political future, which is likely to include a Democratic Party presidential run in 2020. It s reported that Emanuel already has a substantial campaign war chest ready to go, billions of dollars in fact, thanks to some very wealthy backers. If Emanuel does go ahead with the 2020 plan, the driving force behind his campaign will be his Hollywood mogul brother, Ari Emanuel, who will spare no expense in terms of PR, marketing and advertising for his brother s political drive to the White House. Part of the newly found Emanuel family fortune derives from their early investment in the virtual taxi service sensation, Uber, with the Emanuel investment estimated to be valued over $1 billion alone. This is not without controversy however, as it was Mayor Rahm Emanuel s deregulation of Chicago s taxi industry that helped Uber shares to skyrocket in value. Critics are saying that there was a conflict of interest there, but in today s seedy political environment of insider trading and open criminality, it might be hard for any politician to pull rank over that issue. Add to this the potential hundreds of millions more in donations that Emanuel will receive from Jewish property tycoon Sheldon Adelson, and also from Jewish billionaire and Emanuel s fellow anti-gun advocate, Michael Bloomberg   and all in the interests of serving Israel, and you can see how Emanuel could be viewed as a 2020 Democratic front-runner already.The Mayor emerged on to the DC sewer scene as part of the Clinton-Obama Chicago political machine. He s been described by AFP s Victor Thorn as,  More sinister than Karl Rove and potentially deadlier than Dick  Darth Vader  Cheney . Back when Emanuel was Barack Obama s chief of staff in 2008, many commentators remarked on his psychopathic behavior, including his frightening Machiavellian exhibition in a White House meeting where he said the administration s political enemies need to,  Die, die, die! , as he stabbed the table in anger, reportedly using either a pen, or letter opener (regardless, that s a bit scary). Enough to scare the American people considering this is a man who most likely has his eyes on the White House. DUAL LOYALTIES: Rahm Emanuel might have a hard to time putting US interests ahead of Netanyahu s.The other strange bone of contention about Emanuel as a candidate is that he is reportedly a dual citizen (although its believed he gave up his Israeli citizenship at 19 years old) who shares a national identity between the United States and Israel, and actual served in the IDF briefly after college. According to Mondoweiss,  Emanuel left when the Gulf War broke out, in order to volunteer in the IDF. He served in one of Israel s northern bases until the war ended, and upon his return to the US became Clinton s advisor in the White House for almost eight years. Emanuel can often be seen in the presence of Israel leader and Zionist extremist, Bibi Netanyahu, which raises the real prospect of Tel Aviv finally having its very own US president embedded in the White House.The fundamental question here is: when it comes down to the crunch, which country s interest would Emanuel be more loyal towards, the US, or Israel. Contrary to popular Republican belief sets, you cannot wear allegiance to both, because no matter how hard the Israel Lobby tries to portray it   both countries have very radically different national and social interests.If this issue sinks any further down the cesspit of public outrage, expect Emanuel s Hollywood PR machine to go into overdrive.READ MORE ISRAEL NEWS AT: 21st Century Wire Israel Files                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  1\n",
      "Name: text, Length: 17455, dtype: int64\n",
      "\n",
      "Unique values in subject:\n",
      "News               9050\n",
      "politics           6841\n",
      "left-news          4459\n",
      "Government News    1570\n",
      "US_News             783\n",
      "Middle-east         778\n",
      "Name: subject, dtype: int64\n",
      "\n",
      "Unique values in date:\n",
      "May 10, 2017         46\n",
      "May 26, 2016         44\n",
      "May 6, 2016          44\n",
      "May 5, 2016          44\n",
      "May 11, 2016         43\n",
      "                     ..\n",
      "December 9, 2017      1\n",
      "December 4, 2017      1\n",
      "November 19, 2017     1\n",
      "November 20, 2017     1\n",
      "Jul 19, 2015          1\n",
      "Name: date, Length: 1681, dtype: int64\n",
      "\n"
     ]
    }
   ],
   "source": [
    "#Unique Value Counts for Categorical Variables [Fake News]\n",
    "categorical_cols = df_fake.select_dtypes(include=['object']).columns\n",
    "for col in categorical_cols:\n",
    "    print(f\"Unique values in {col}:\")\n",
    "    print(df_fake[col].value_counts())\n",
    "    print()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "68b6634b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Unique values in title:\n",
      "Factbox: Trump fills top jobs for his administration                                14\n",
      "Factbox: Contenders for senior jobs in Trump's administration                        8\n",
      "Highlights: The Trump presidency on April 13 at 9:30 P.M. EDT/0130 GMT on Friday     8\n",
      "Factbox: International reaction to arrest of Reuters reporters in Myanmar            6\n",
      "Highlights: The Trump presidency on April 21 at 6:12 p.m. EDT/2212 GMT               5\n",
      "                                                                                    ..\n",
      "Obama, on last trip to Europe, warns against nationalism, populism                   1\n",
      "A post-Trump SEC could shake up current policy                                       1\n",
      "U.S. panel urges probe on whether China weakening U.S. militarily                    1\n",
      "Trump team weighs 'infrastructure bank' to fund projects: Trump adviser              1\n",
      "Indonesia to buy $1.14 billion worth of Russian jets                                 1\n",
      "Name: title, Length: 20826, dtype: int64\n",
      "\n",
      "Unique values in text:\n",
      "(Reuters) - Highlights for U.S. President Donald Trump’s administration on Thursday: The United States drops a massive GBU-43 bomb, the largest non-nuclear bomb it has ever used in combat, in Afghanistan against a series of caves used by Islamic State militants, the Pentagon says. Trump says Pyongyang is a problem that “will be taken care of” amid speculation that North Korea is on the verge of a sixth nuclear test. Military force cannot resolve tension over North Korea, China warns, while an influential Chinese newspaper urges Pyongyang to halt its nuclear program in exchange for Beijing’s protection. The Trump administration is focusing its North Korea strategy on tougher economic sanctions, possibly including intercepting cargo ships and punishing Chinese banks doing business with Pyongyang, U.S. officials say. Trump says “things will work out fine” between the United States and Russia, a day after declaring U.S.-Russian relations may be at an all-time low. Trump signals he could be moving closer to the mainstream on monetary policy, saying he has not ruled out reappointment of Janet Yellen as Federal Reserve chair as he considers his choices for the U.S. central bank. [nL1N1HL14B] Trump signs a resolution that will allow U.S. states to restrict how federal funds for contraception and reproductive health are spent, a move cheered by anti-abortion campaigners. Democratic Senator Chris Van Hollen presses Deutsche Bank to release information about issues including Trump’s debt and any bank meetings with Trump administration officials, saying he has “great concern” about possible conflicts of interest. EXPORT-IMPORT BANK Trump’s office says he plans to revive the hobbled Export-Import Bank of the United States, a victory for American manufacturers such as Boeing Co and General Electric Co that have overseas customers that use the agency’s government-backed loans to purchase their products. Top Wall Street bankers say they are having positive discussions about financial regulation in Washington, and downplay the idea U.S. policymakers may force their institutions to split up. The United States is pushing for trade to be a key issue in top-level economic talks with Japan, a source says, an unwelcome development for Tokyo, which is seeking to fend off U.S. pressure to reduce the bilateral trade imbalance. Trump’s administration has focused on one group of illegal immigrants more than others: women with children, according to eight Department of Homeland Security officials interviewed by Reuters about agency planning.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         8\n",
      "(Reuters) - Highlights for U.S. President Donald Trump’s administration on Friday: Trump promises a big announcement about tax reform next week and orders an administration review of Obama-era tax rules written to discourage U.S. companies from relocating overseas to cut their tax bills. Trump tells the Treasury Department to examine two powers given to regulators to police large financial companies following the 2008 financial crisis. South Korea says it is on heightened alert ahead of another important anniversary in North Korea, with a large concentration of military hardware amassed on both sides of the border amid concerns about a new nuclear test by Pyongyang. Trump, striving to make good on a top campaign promise, is pushing fellow Republicans who control Congress to pass revamped healthcare legislation but the same intraparty squabbling that torpedoed it last month could do it again. Defense Secretary Jim Mattis says Syria has dispersed its warplanes in recent days and that it retains chemical weapons, an issue he says will have to be taken up diplomatically. The Department of Justice threatens to cut off funding to California as well as eight cities and counties across the United States, escalating a Trump administration crackdown on so-called “sanctuary cities” that do not cooperate with federal immigration authorities. The United States will not make an exception for American companies, including oil major Exxon Mobil Corp, seeking to drill in areas prohibited by U.S. sanctions on Russia, Treasury Secretary Steven Mnuchin says.  Trump and his fellow Republicans who control Congress face their first major budget test next week, with the threat of a government shutdown potentially hinging on his proposed Mexican border wall as well as Obamacare funding. The House of Representatives Intelligence Committee says it has invited FBI, National Security Agency and Obama administration officials to testify as it restarts its investigation into alleged Russian meddling in the 2016 U.S. election. U.N. Secretary-General Antonio Guterres meets with Trump at the White House for the first time since both took office earlier this year and amid a U.S. push to cut funding to the world body and its agencies. The United States has offered to help fund Mexico’s efforts to eradicate opium poppies, a U.S. official says, as Mexican heroin output increased again last year.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       5\n",
      "(Reuters) - Highlights of the day for U.S. President Donald Trump’s administration on Friday: Trump backs a decision by his former national security adviser, Michael Flynn, to seek immunity in congressional investigations of possible ties between his campaign and Russia, but there is no immediate sign the request will be granted.  The Trump administration slams China on a range of trade issues from its chronic industrial overcapacity to forced technology transfers and longstanding bans on U.S. beef and electronic payment services. Beijing seeks to play down tensions with the United States and put on a positive face ahead of President Xi Jinping’s first meeting with Trump next week. Senate Democrats step closer to having enough votes to block a confirmation vote on Trump’s Supreme Court nominee with three more Democratic senators coming out against Neil Gorsuch for the lifetime job as a justice. Trump seeks to push his plan for fair trade and more manufacturing jobs back to the top of his agenda by ordering a study into the causes of U.S. trade deficits and a clampdown on import duty evasion. Trump has neither a clear White House tax plan nor adequate staff yet to see through a planned tax overhaul, according to interviews with people in the administration, in Congress and among U.S. tax experts. Democrats are trying to counter Trump’s boldest move yet to defang the U.S. consumer financial watchdog, with 40 current and former lawmakers defending the agency in court. The U.S. Environmental Protection Agency’s scientific integrity watchdog is reviewing whether EPA chief Scott Pruitt violated the agency’s policies when he said in a television interview he does not believe carbon dioxide is driving global climate change, according to an email seen by Reuters. Trump will seek to rebuild the U.S. relationship with Egypt at a meeting on Monday with Egyptian President Abdel Fattah al-Sisi focused on security issues and military aid, a senior White House official says. Trump will host Jordan’s King Abdullah at the White House next week to discuss the fight against Islamic State militants, the Syria crisis and advancing peace between Israelis and Palestinians, the White House says. A U.S. judge approves a $25 million settlement to resolve a class action lawsuit that claimed fraud against Trump and his Trump University real estate seminars.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     5\n",
      "(Reuters) - Highlights for U.S. President Donald Trump’s administration on Wednesday: Trump proposes slashing tax rates for businesses and on overseas corporate profits returned to the country in a plan greeted as an opening gambit by his fellow Republicans in Congress. Trump’s plan could shift the U.S. economy into higher gear but could have one effect the White House would not welcome — interest rates ratcheted higher than expected by a wary central bank. The Trump tax cut will generate growth but not nearly enough to replace trillions of dollars in lost revenues, while rising deficits could even take back some of the economic gains, fiscal experts say. Congress inches toward a deal to fund the government through September but is preparing to possibly extend a midnight Friday deadline in order to wrap up negotiations and avoid an imminent government shutdown. Trump is considering issuing an executive order to pull the United States from the North American Free Trade Agreement, an administration official says, a move that could unravel one of the world’s biggest trading blocs. Trump and Canadian Prime Minister Justin Trudeau discuss bilateral trade in their second conversation in as many days  amid strains over softwood lumber and dairy. The Trump administration says it aims to push North Korea into dismantling its nuclear and missile programs through tougher international sanctions and diplomatic pressure, and remains open to negotiations to bring that about. Trump gives the military the authority to reset a confusing system of troop limits in Iraq and Syria that critics say allows the White House to micro-manage battlefield decisions and ultimately obscures the real number of U.S. forces. Trump signs an executive order to allow national monument designations to be rescinded or reduce the size of sites as the administration pushes to open more federal land to drilling, mining and other development. Trump orders Education Secretary Betsy DeVos to review the government role in school policy, which supporters cheer as a step in creating more local control in education and critics worry it could lead to lower quality schools in poorer neighborhoods. Israel’s intelligence minister says his country wants an “understanding” with the Trump administration that Iran must not be allowed to establish a permanent military foothold in Syria.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    3\n",
      "BANGKOK (Reuters) - Thailand on Wednesday marked the start of a lavish, five-day funeral for King Bhumibol Adulyadej with a Buddhist religious ceremony attended by senior members of its royal family. King Bhumibol, who died last year aged 88, will be cremated on Thursday on a royal pyre within a cremation complex of gold pavilions in front of Bangkok s Grand Palace, in a ceremony that is expected to draw about 250,000 mourners. Thailand has observed a year of mourning for King Bhumibol, who was regarded as a pillar of stability during a reign of seven decades that witnessed political upheaval and rapid development in the Southeast Asian nation.   It s overwhelming,  said one mourner, Aporn Wongdee, 60,  who hails from the southern province of Nakhon Si Thammarat.  I ve been here for two days already and I want to see our father to heaven.       A sum of $90 million has been set aside for the funeral, the likes of which has never been seen in Thailand, officials involved in the funeral preparations said. King Maha Vajiralongkorn, known as Rama X, who inherited the throne in December on his father s death, arrived at the Grand Palace by car on Wednesday as soldiers dressed in red uniforms and black hats stood to attention.  He was flanked by his two daughters. Live television images from inside the palace showed the king lighting candles in front of his father s coffin and a symbolic royal urn. The Buddhist funeral ceremony, mixed with Hindu rituals, was attended by 119 Buddhist monks who chanted prayers in the ancient Pali language. Queues of black-clad mourners, many carrying portraits of the king, snaked around parts of Bangkok s old town, waiting to enter the cremation area. By mid-afternoon, 25,000 mourners had gathered around the cremation site, city police said. In what is expected to be an emotionally-charged morning, King Bhumibol s body will be moved on Thursday from the Grand Palace to a crematorium in a public square in front, where thousands of people have already pitched tents to ensure places. On Thursday, three processions will make their way from the palace to the cremation site - a series of specially-erected Thai pavilions that took nearly a year to build. Some Thais have folded flowers of sandalwood paper to be used in the cremation, in the belief that their fragrance guides the soul of the departed to heaven. The cremation day has been declared a national holiday, when banks will be closed and major shopping centers will be shut from 3 p.m.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                2\n",
      "                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      ..\n",
      "WASHINGTON (Reuters) - President-elect Donald Trump’s shortlist of candidates to lead the U.S. Department of Interior has employees and environmental advocates fearful of a shift in the agency’s direction, from one focused on preserving public lands to one that would open them up to more drilling and mining. The outcome will have implications for industry access to millions of acres of national parks, reserves and tribal territories stretching from the Arctic to the Gulf of Mexico and the viability of President Barack Obama’s efforts to keep the United States in line with international agreements to reduce the impacts of climate change. Republican Trump, a New York real estate businessman who has never previously held public office, has leaned toward right-wing loyalists for the Cabinet since winning the Nov. 8 election. He is considering oil drilling advocates like Oklahoma Governor Mary Fallin, Alaska’s former governor Sarah Palin and Lucas Oil co-founder Forrest Lucas to run the Interior Department, media reports and Reuters sources said. Other contenders are several politicians from Western states who favor easier development of public lands. Any of those picks could trigger battles with environmental groups and cause internal strife at an agency where many workers see themselves as land stewards after nearly eight years of conservation-minded policies under Democrat Obama. “Public lands have been set aside to ‘preserve and protect’ cultural and scientific resources for future generations,” said Geoff Goins, a National Park Service ranger at the Bandelier National Monument in New Mexico, adding that with Trump coming in, “people are concerned about their jobs.”   Other Interior Department employees interviewed by Reuters said they were worried the agency’s environmental mandate would be weakened under Trump, and green advocates said they were bracing to resist those changes. “Climate change is a major focus of conservation concern for national parks,” said one National Park Service employee in the Northwest who asked not to be identified. “If (Trump’s administration) gets in the way of scientists...we are all in deep trouble.” During the election campaign, Trump tweeted that “the concept of global warming was created by and for the Chinese in order to make U.S. manufacturing non-competitive” - a view that is at odds with most but a few scientists who study the impacts of rising global temperatures and extreme weather.        Maureen Finnerty, chair of the Coalition to Protect America’s National Parks, an organization of more than 1,200 current and former National Parks employees, said it was ready to launch a public relations campaign against Trump if he pursues an anti-environmental agenda. “We will be vigilant and hope for the best,” she said. The Interior Department employs more than 70,000 people across the United States and oversees over 20 percent of federal land. Under Obama, the Interior Department played a big role in efforts to curb the effects of climate change by limiting fossil fuel development in some areas. Outgoing Interior Secretary Sally Jewell banned coal mining on public lands, canceled leases for drilling in the Arctic and Atlantic coasts, expanded wildlife protection and cracked down on industry methane emissions. The Obama administration planned on using forest restoration on federal lands as a way to help the United States meet its long-term goals under the 2015 Paris agreement within the United Nations Framework Convention on Climate Change. The agreement outlines how countries will deal with lowering greenhouse gas emissions starting in 2020.  Trump has given mixed messages on his plans for Interior. In an interview with Field and Stream magazine in January, Trump said: “I want to keep the lands great... We have to be great stewards of this land.” But he has advocated strongly for boosting energy development on federal lands and has criticized Obama’s use of environmental regulation to check oil and gas development. He picked renowned climate change skeptic Myron Ebell to run his transition at the Environmental Protection Agency. U.S. Representative Kevin Cramer of North Dakota said Trump could consider hiring as his energy secretary Harold Hamm, an oil and gas driller and a pioneer of developing shale oil resources. One potential Interior Department head is Oklahoma Governor Fallin, who met with Trump on Monday. She has been an ardent supporter of Oklahoma’s drilling industry and has blocked attempts to ban hydraulic fracturing, or fracking, a controversial drilling technology. Fallin’s spokesman confirmed she is being considered for the post, but said there has been “no offer given.” Also on the shortlist is Palin, who made famous the motto “Drill, Baby, Drill” when she was the vice presidential running-mate to Republican John McCain in 2008, and former Arizona Governor Jan Brewer, a Trump supporter without experience in public lands policy. Governor Butch Otter of Idaho, venture capitalist Robert Grady and U.S. Representatives Cynthia Lummis and Rob Bishop of Wyoming and Utah are also potential candidates for the job. All declined comment. (Refiles to correct typographical error National Park Service instead of National Parks Service.)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     1\n",
      "BEDMINSTER, N.J. (Reuters) - President-elect Donald Trump on Sunday assessed several more contenders for top U.S. posts including Chris Christie and Rudy Giuliani, as blunt-spoken retired Marine Corps General James Mattis emerged as a leading candidate for defense secretary. Trump held meetings at his golf resort in Bedminster, New Jersey, with candidates for senior administration jobs after he takes office on Jan. 20. On Saturday, he conferred with Mattis and Mitt Romney, formerly a fierce Trump critic now under consideration for secretary of state. Summing up two days of talks as he said goodbye to retired U.S. Marines Corps General John Kelly on Sunday evening, Trump said he had made decisions on a couple of appointments. “We really had some great meetings, and you’ll be hearing about them soon.” Trump transition spokesman Sean Spicer said no decisions would be announced on Sunday night. Trump discussed the treasury secretary job with Jon Gray, a board member of the Blackstone Group, two sources familiar with their meeting said. A Trump transition team statement said their discussion included the U.S. economy, global capital markets and the world financial situation, as well as tax reform and long-term debt. The last person Trump escorted out of the clubhouse was Giuliani, the former New York mayor. Trump said earlier that Giuliani was a candidate for secretary of state “and other things.” The Trump team statement said Trump and Giuliani discussed “administration priorities” as well as “restoring America’s prominence in the world, ongoing national security issues and threats at various hotspots on a global basis.” Christie, the Republican governor of New Jersey, advised Trump during the presidential campaign but was dismissed as the head of his transition team. Asked by reporters before the meeting whether there was a place for Christie in his administration, Trump sidestepped the question but called him “a very talented man, great guy.” Trump met with billionaire investor Wilbur Ross, who he said was under consideration for commerce secretary. Asked whether he wanted the job, Ross told reporters: “Well, time will tell.”  Trump also received Jonathan Gray, the global head of real estate at the Blackstone Group, who is being considered for Treasury secretary, according to a person briefed on the matter. The president-elect has already tapped three senior leaders of his national security and law enforcement teams, choosing U.S. Senator Jeff Sessions for attorney general, U.S. Representative Mike Pompeo as CIA director, and retired Lieutenant General Michael Flynn as national security adviser. The selections so far suggest that Trump, true to his campaign promises, intends to steer national security and foreign policy in a sharply different direction from that of Democratic President Barack Obama, whose record Trump harshly criticized during his campaign. Obama campaigned extensively for Democratic presidential nominee Hillary Clinton before the Nov. 8 election and warned during the campaign that Trump lacked the temperament and qualifications to be president. In Peru, as he wrapped up his last foreign trip, one that was overshadowed by Trump’s surprise election, Obama encouraged Democrats on Sunday to work with the new administration but said he would weigh in as a citizen if the Trump administration challenged values he holds dear.  In another sign of Trump’s novel approach to politics, the New York businessman who has never held public office said he planned to live in the White House but that his wife, Melania, and their 10-year-old son, Barron, would not move in immediately. He said they would move from New York to the White House “very soon, right after he finishes school.” While Trump received lawmakers, politicians and business leaders behind closed doors over the weekend, he gave some hints about his thinking and possible Cabinet choices on Twitter.  On Sunday, he tweeted that “General James ‘Mad Dog’ Mattis, who is being considered for Secretary of Defense, was very impressive yesterday. A true General’s General!” From 2010 to 2013, Mattis headed the U.S. military’s Central Command, which oversees operations stretching from the Horn of Africa through the Middle East and into Central Asia including Afghanistan and Pakistan. During that time, he was at odds with the Obama administration on the need to prepare for potential threats from Iran and about resources for Afghanistan. Mattis, 66, served as an American commander in the wars in Iraq and Afghanistan and was known to be popular among the troops. Speaking on “Fox News Sunday,” Vice President-elect Mike Pence, who now heads Trump’s transition team,” said Mattis had “a legendary military career.” Incoming White House Chief of Staff Reince Priebus said it was a “very real possibility” Mattis would get the job, telling the ABC program “This Week:” “I know that President-elect Trump loves leaders like General Mattis.” Pence said that Trump and Romney had a good meeting and “a warm and a substantive exchange.” “I can say that Governor Romney is under active and serious consideration to serve as secretary of state of the United States,” Pence said on the CBS program “Face the Nation.”    Romney, the unsuccessful 2012 Republican presidential nominee, was a leader of the Republican establishment movement that tried to block Trump from becoming the nominee this year. In March, Romney called Trump “a phony,” “a fraud” and “a con man.” A source close to Romney from his time as Massachusetts governor expressed concern he might be “frozen out” by officials whose thinking appears to be closer to Trump’s, such as Flynn, Mattis, White House counselor Steve Bannon, and members of Trump’s family. “How much influence and latitude he will have will be up to Trump, and they don’t appear to be on the same page about much,” the source said. Former Texas Governor Rick Perry, who made unsuccessful bids for the Republican presidential nomination in 2012 and 2016, will meet with Trump on Monday and is being considered for Cabinet posts including defense, energy and veterans affairs, Trump’s transition team said.     1\n",
      "NEW YORK (Reuters) - U.S. President-elect Donald Trump’s choice of experts to focus on new policies at the Federal Communications Commission signals a regime that will have a “lighter” touch on regulation and be more likely to favor large mergers in telecoms industries, analysts said. Economist Jeff Eisenach and former Sprint Corp (S.N) lobbyist Mark Jamison were named by Trump’s transition team to oversee hiring and policy for the FCC. They both oppose some recent telecom industry regulations resisted by telecom and cable heavyweights such as Comcast Corp (CMCSA.O) and AT&T Inc (T.N) and have voiced support for mega mergers in the past. The FCC is composed of five commissioners, including one designated as chairman, who are appointed by the president and confirmed by the Senate. Only three commissioners can be from the same political party and Trump’s pick for FCC chairman would tip the balance in favor of Republicans. The addition of Eisenach and Jamison to Trump’s “landing team” on Monday come as the Republican president-elect puts together a team to staff various government departments and agencies after he succeeds Democratic President Barack Obama on Jan. 20. The two appointments are harbingers of “a more typical Republican FCC that is lighter on regulation and more focused on competition,” said Roger Entner, an analyst at Recon Analytics. “The focus will be more on reducing regulation than creating new ones.” That would be in stark contrast to the Obama administration’s FCC that enacted or proposed a handful of new industry rules and disapproved some proposed combinations, including Comcast’s bid for Time Warner Cable and AT&T’s attempt to buy T-Mobile (TMUS.O).The FCC under Chairman Tom Wheeler, a Democrat, has had a rocky relationship with large telecom companies, some of which strongly opposed the agency’s 2015 net neutrality or open internet rules. The rules, which require internet service providers to treat all data equally and bar them from obstructing or slowing down consumer access to web content, were seen as a major victory for internet businesses like Alphabet Inc’s Google (GOOGL.O) that offer services but do not own internet networks. ‘PRO-BUSINESS POLICY’ Eisenach has supported mergers such as AT&T and T-Mobile as well as Sprint and T-Mobile that were dismissed by regulators during Obama’s administration, according to analysts. He is known in telecom circles for having a “pro-business” mindset, New Street Research analyst Spencer Kurn said.  “Whoever gets picked (as FCC chairman) is likely going to implement a similar pro-business policy,” Kurn said.  Eisenach, who was previously tapped by the Trump campaign as an adviser on technology and telecom policy, is currently a managing director at consulting firm NERA Economic Consulting’s communications, media and internet practice. He previously held advisory roles at the U.S. Federal Trade Commission and the White House Office of Management and Budget. In 2014, he testified in the U.S. Senate against net neutrality rules, arguing there was no need for new regulation as existing antitrust rules, while “not perfect,” offered safeguards against concerns about the business practices of internet service providers. Jamison is the director of the Public Utility Research Center at the University of Florida. Like Eisenach, he has strongly disagreed in his publications with Wheeler’s rules, including net neutrality and a proposal to open up the market for rented pay-TV set-top boxes. That measure was aimed at breaking the telecom and cable companies’ grip on the $20 billion market and bringing in players such as Google and Apple (AAPL.O) in an effort to lower prices for consumers. Some analysts said the appointments also raised questions over whether Trump would carry out his campaign pledge to kill AT&T’s $85.4 billion proposal to buy Time Warner (TWX.N). Matt Wood, policy director of the technology rights group Free Press, said that even as Trump opposes the deal, he picked the pro-business economist Eisenach, who was unlikely to want to block such a merger.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            1\n",
      "WASHINGTON (Reuters) - U.S. President-elect Donald Trump released a video on Monday laying out actions he will take on his first day in office on Jan. 20, including withdrawing the United States from a Trans-Pacific Partnership trade deal. Trump also said he would issue a rule cutting government regulations, direct the Labor Department to investigate abuses of visa programs, and cancel some restrictions on energy production, including shale oil and gas and coal.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     1\n",
      "JAKARTA (Reuters) - Indonesia will buy 11 Sukhoi fighter jets worth $1.14 billion from Russia in exchange for cash and Indonesian commodities, two cabinet ministers said on Tuesday. The Southeast Asian country has pledged to ship up to $570 million worth of commodities in addition to cash to pay for the Suhkoi SU-35 fighter jets, which are expected to be delivered in stages starting in two years. Indonesian Trade Minister Enggartiasto Lukita said in a joint statement with Defence Minister Ryamizard Ryacudu that details of the type and volume of commodities were  still being negotiated . Previously he had said the exports could include palm oil, tea, and coffee. The deal is expected to be finalised soon between Indonesian state trading company PT Perusahaan Perdangangan Indonesia and Russian state conglomerate Rostec. Russia is currently facing a new round of U.S.-imposed trade sanctions. Meanwhile, Southeast Asia s largest economy is trying to promote its palm oil products amid threats of a cut in consumption by European Union countries. Indonesia is also trying to modernize its ageing air force after a string of military aviation accidents. Indonesia, which had a $411 million trade surplus with Russia in 2016, wants to expand bilateral cooperation in tourism, education, energy, technology and aviation among others.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              1\n",
      "Name: text, Length: 21192, dtype: int64\n",
      "\n",
      "Unique values in subject:\n",
      "politicsNews    11272\n",
      "worldnews       10145\n",
      "Name: subject, dtype: int64\n",
      "\n",
      "Unique values in date:\n",
      "December 20, 2017      182\n",
      "December 6, 2017       166\n",
      "November 30, 2017      162\n",
      "November 9, 2017       158\n",
      "October 13, 2017       155\n",
      "                      ... \n",
      "August 6, 2016           1\n",
      "August 21, 2016          1\n",
      "September 3, 2016        1\n",
      "September 11, 2016       1\n",
      "January 24, 2016         1\n",
      "Name: date, Length: 716, dtype: int64\n",
      "\n"
     ]
    }
   ],
   "source": [
    "#Unique Value Counts for Categorical Variables [True News]\n",
    "categorical_cols = df_true.select_dtypes(include=['object']).columns\n",
    "for col in categorical_cols:\n",
    "    print(f\"Unique values in {col}:\")\n",
    "    print(df_true[col].value_counts())\n",
    "    print()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "c4bc346e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Removing last 10 rows for manual testing\n",
    "df_fake_manual_testing = df_fake.tail(10)\n",
    "for i in range(23480,23470,-1):\n",
    "    df_fake.drop([i], axis = 0, inplace = True)\n",
    "    \n",
    "df_true_manual_testing = df_true.tail(10)\n",
    "for i in range(21416,21406,-1):\n",
    "    df_true.drop([i], axis = 0, inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "7e9ad777",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((21407, 5), (23471, 5))"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_true.shape, df_fake.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "ff7928cc",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\hp\\AppData\\Local\\Temp\\ipykernel_23124\\860779283.py:1: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  df_fake_manual_testing[\"class\"] = 0\n",
      "C:\\Users\\hp\\AppData\\Local\\Temp\\ipykernel_23124\\860779283.py:2: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  df_true_manual_testing[\"class\"] = 1\n"
     ]
    }
   ],
   "source": [
    "df_fake_manual_testing[\"class\"] = 0\n",
    "df_true_manual_testing[\"class\"] = 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "525a4bfd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>title</th>\n",
       "      <th>text</th>\n",
       "      <th>subject</th>\n",
       "      <th>date</th>\n",
       "      <th>class</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>23471</th>\n",
       "      <td>Seven Iranians freed in the prisoner swap have...</td>\n",
       "      <td>21st Century Wire says This week, the historic...</td>\n",
       "      <td>Middle-east</td>\n",
       "      <td>January 20, 2016</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23472</th>\n",
       "      <td>#Hashtag Hell &amp; The Fake Left</td>\n",
       "      <td>By Dady Chery and Gilbert MercierAll writers ...</td>\n",
       "      <td>Middle-east</td>\n",
       "      <td>January 19, 2016</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23473</th>\n",
       "      <td>Astroturfing: Journalist Reveals Brainwashing ...</td>\n",
       "      <td>Vic Bishop Waking TimesOur reality is carefull...</td>\n",
       "      <td>Middle-east</td>\n",
       "      <td>January 19, 2016</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23474</th>\n",
       "      <td>The New American Century: An Era of Fraud</td>\n",
       "      <td>Paul Craig RobertsIn the last years of the 20t...</td>\n",
       "      <td>Middle-east</td>\n",
       "      <td>January 19, 2016</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23475</th>\n",
       "      <td>Hillary Clinton: ‘Israel First’ (and no peace ...</td>\n",
       "      <td>Robert Fantina CounterpunchAlthough the United...</td>\n",
       "      <td>Middle-east</td>\n",
       "      <td>January 18, 2016</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23476</th>\n",
       "      <td>McPain: John McCain Furious That Iran Treated ...</td>\n",
       "      <td>21st Century Wire says As 21WIRE reported earl...</td>\n",
       "      <td>Middle-east</td>\n",
       "      <td>January 16, 2016</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23477</th>\n",
       "      <td>JUSTICE? Yahoo Settles E-mail Privacy Class-ac...</td>\n",
       "      <td>21st Century Wire says It s a familiar theme. ...</td>\n",
       "      <td>Middle-east</td>\n",
       "      <td>January 16, 2016</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23478</th>\n",
       "      <td>Sunnistan: US and Allied ‘Safe Zone’ Plan to T...</td>\n",
       "      <td>Patrick Henningsen  21st Century WireRemember ...</td>\n",
       "      <td>Middle-east</td>\n",
       "      <td>January 15, 2016</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23479</th>\n",
       "      <td>How to Blow $700 Million: Al Jazeera America F...</td>\n",
       "      <td>21st Century Wire says Al Jazeera America will...</td>\n",
       "      <td>Middle-east</td>\n",
       "      <td>January 14, 2016</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23480</th>\n",
       "      <td>10 U.S. Navy Sailors Held by Iranian Military ...</td>\n",
       "      <td>21st Century Wire says As 21WIRE predicted in ...</td>\n",
       "      <td>Middle-east</td>\n",
       "      <td>January 12, 2016</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                   title  \\\n",
       "23471  Seven Iranians freed in the prisoner swap have...   \n",
       "23472                      #Hashtag Hell & The Fake Left   \n",
       "23473  Astroturfing: Journalist Reveals Brainwashing ...   \n",
       "23474          The New American Century: An Era of Fraud   \n",
       "23475  Hillary Clinton: ‘Israel First’ (and no peace ...   \n",
       "23476  McPain: John McCain Furious That Iran Treated ...   \n",
       "23477  JUSTICE? Yahoo Settles E-mail Privacy Class-ac...   \n",
       "23478  Sunnistan: US and Allied ‘Safe Zone’ Plan to T...   \n",
       "23479  How to Blow $700 Million: Al Jazeera America F...   \n",
       "23480  10 U.S. Navy Sailors Held by Iranian Military ...   \n",
       "\n",
       "                                                    text      subject  \\\n",
       "23471  21st Century Wire says This week, the historic...  Middle-east   \n",
       "23472   By Dady Chery and Gilbert MercierAll writers ...  Middle-east   \n",
       "23473  Vic Bishop Waking TimesOur reality is carefull...  Middle-east   \n",
       "23474  Paul Craig RobertsIn the last years of the 20t...  Middle-east   \n",
       "23475  Robert Fantina CounterpunchAlthough the United...  Middle-east   \n",
       "23476  21st Century Wire says As 21WIRE reported earl...  Middle-east   \n",
       "23477  21st Century Wire says It s a familiar theme. ...  Middle-east   \n",
       "23478  Patrick Henningsen  21st Century WireRemember ...  Middle-east   \n",
       "23479  21st Century Wire says Al Jazeera America will...  Middle-east   \n",
       "23480  21st Century Wire says As 21WIRE predicted in ...  Middle-east   \n",
       "\n",
       "                   date  class  \n",
       "23471  January 20, 2016      0  \n",
       "23472  January 19, 2016      0  \n",
       "23473  January 19, 2016      0  \n",
       "23474  January 19, 2016      0  \n",
       "23475  January 18, 2016      0  \n",
       "23476  January 16, 2016      0  \n",
       "23477  January 16, 2016      0  \n",
       "23478  January 15, 2016      0  \n",
       "23479  January 14, 2016      0  \n",
       "23480  January 12, 2016      0  "
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_fake_manual_testing.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "14ad751a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>title</th>\n",
       "      <th>text</th>\n",
       "      <th>subject</th>\n",
       "      <th>date</th>\n",
       "      <th>class</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>21407</th>\n",
       "      <td>Mata Pires, owner of embattled Brazil builder ...</td>\n",
       "      <td>SAO PAULO (Reuters) - Cesar Mata Pires, the ow...</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>August 22, 2017</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>21408</th>\n",
       "      <td>U.S., North Korea clash at U.N. forum over nuc...</td>\n",
       "      <td>GENEVA (Reuters) - North Korea and the United ...</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>August 22, 2017</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>21409</th>\n",
       "      <td>U.S., North Korea clash at U.N. arms forum on ...</td>\n",
       "      <td>GENEVA (Reuters) - North Korea and the United ...</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>August 22, 2017</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>21410</th>\n",
       "      <td>Headless torso could belong to submarine journ...</td>\n",
       "      <td>COPENHAGEN (Reuters) - Danish police said on T...</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>August 22, 2017</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>21411</th>\n",
       "      <td>North Korea shipments to Syria chemical arms a...</td>\n",
       "      <td>UNITED NATIONS (Reuters) - Two North Korean sh...</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>August 21, 2017</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>21412</th>\n",
       "      <td>'Fully committed' NATO backs new U.S. approach...</td>\n",
       "      <td>BRUSSELS (Reuters) - NATO allies on Tuesday we...</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>August 22, 2017</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>21413</th>\n",
       "      <td>LexisNexis withdrew two products from Chinese ...</td>\n",
       "      <td>LONDON (Reuters) - LexisNexis, a provider of l...</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>August 22, 2017</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>21414</th>\n",
       "      <td>Minsk cultural hub becomes haven from authorities</td>\n",
       "      <td>MINSK (Reuters) - In the shadow of disused Sov...</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>August 22, 2017</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>21415</th>\n",
       "      <td>Vatican upbeat on possibility of Pope Francis ...</td>\n",
       "      <td>MOSCOW (Reuters) - Vatican Secretary of State ...</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>August 22, 2017</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>21416</th>\n",
       "      <td>Indonesia to buy $1.14 billion worth of Russia...</td>\n",
       "      <td>JAKARTA (Reuters) - Indonesia will buy 11 Sukh...</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>August 22, 2017</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                   title  \\\n",
       "21407  Mata Pires, owner of embattled Brazil builder ...   \n",
       "21408  U.S., North Korea clash at U.N. forum over nuc...   \n",
       "21409  U.S., North Korea clash at U.N. arms forum on ...   \n",
       "21410  Headless torso could belong to submarine journ...   \n",
       "21411  North Korea shipments to Syria chemical arms a...   \n",
       "21412  'Fully committed' NATO backs new U.S. approach...   \n",
       "21413  LexisNexis withdrew two products from Chinese ...   \n",
       "21414  Minsk cultural hub becomes haven from authorities   \n",
       "21415  Vatican upbeat on possibility of Pope Francis ...   \n",
       "21416  Indonesia to buy $1.14 billion worth of Russia...   \n",
       "\n",
       "                                                    text    subject  \\\n",
       "21407  SAO PAULO (Reuters) - Cesar Mata Pires, the ow...  worldnews   \n",
       "21408  GENEVA (Reuters) - North Korea and the United ...  worldnews   \n",
       "21409  GENEVA (Reuters) - North Korea and the United ...  worldnews   \n",
       "21410  COPENHAGEN (Reuters) - Danish police said on T...  worldnews   \n",
       "21411  UNITED NATIONS (Reuters) - Two North Korean sh...  worldnews   \n",
       "21412  BRUSSELS (Reuters) - NATO allies on Tuesday we...  worldnews   \n",
       "21413  LONDON (Reuters) - LexisNexis, a provider of l...  worldnews   \n",
       "21414  MINSK (Reuters) - In the shadow of disused Sov...  worldnews   \n",
       "21415  MOSCOW (Reuters) - Vatican Secretary of State ...  worldnews   \n",
       "21416  JAKARTA (Reuters) - Indonesia will buy 11 Sukh...  worldnews   \n",
       "\n",
       "                   date  class  \n",
       "21407  August 22, 2017       1  \n",
       "21408  August 22, 2017       1  \n",
       "21409  August 22, 2017       1  \n",
       "21410  August 22, 2017       1  \n",
       "21411  August 21, 2017       1  \n",
       "21412  August 22, 2017       1  \n",
       "21413  August 22, 2017       1  \n",
       "21414  August 22, 2017       1  \n",
       "21415  August 22, 2017       1  \n",
       "21416  August 22, 2017       1  "
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_true_manual_testing.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "b1f0336d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>title</th>\n",
       "      <th>text</th>\n",
       "      <th>subject</th>\n",
       "      <th>date</th>\n",
       "      <th>class</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Donald Trump Sends Out Embarrassing New Year’...</td>\n",
       "      <td>Donald Trump just couldn t wish all Americans ...</td>\n",
       "      <td>News</td>\n",
       "      <td>December 31, 2017</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Drunk Bragging Trump Staffer Started Russian ...</td>\n",
       "      <td>House Intelligence Committee Chairman Devin Nu...</td>\n",
       "      <td>News</td>\n",
       "      <td>December 31, 2017</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Sheriff David Clarke Becomes An Internet Joke...</td>\n",
       "      <td>On Friday, it was revealed that former Milwauk...</td>\n",
       "      <td>News</td>\n",
       "      <td>December 30, 2017</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Trump Is So Obsessed He Even Has Obama’s Name...</td>\n",
       "      <td>On Christmas day, Donald Trump announced that ...</td>\n",
       "      <td>News</td>\n",
       "      <td>December 29, 2017</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Pope Francis Just Called Out Donald Trump Dur...</td>\n",
       "      <td>Pope Francis used his annual Christmas Day mes...</td>\n",
       "      <td>News</td>\n",
       "      <td>December 25, 2017</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>Racist Alabama Cops Brutalize Black Boy While...</td>\n",
       "      <td>The number of cases of cops brutalizing and ki...</td>\n",
       "      <td>News</td>\n",
       "      <td>December 25, 2017</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>Fresh Off The Golf Course, Trump Lashes Out A...</td>\n",
       "      <td>Donald Trump spent a good portion of his day a...</td>\n",
       "      <td>News</td>\n",
       "      <td>December 23, 2017</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>Trump Said Some INSANELY Racist Stuff Inside ...</td>\n",
       "      <td>In the wake of yet another court decision that...</td>\n",
       "      <td>News</td>\n",
       "      <td>December 23, 2017</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>Former CIA Director Slams Trump Over UN Bully...</td>\n",
       "      <td>Many people have raised the alarm regarding th...</td>\n",
       "      <td>News</td>\n",
       "      <td>December 22, 2017</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>WATCH: Brand-New Pro-Trump Ad Features So Muc...</td>\n",
       "      <td>Just when you might have thought we d get a br...</td>\n",
       "      <td>News</td>\n",
       "      <td>December 21, 2017</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                               title  \\\n",
       "0   Donald Trump Sends Out Embarrassing New Year’...   \n",
       "1   Drunk Bragging Trump Staffer Started Russian ...   \n",
       "2   Sheriff David Clarke Becomes An Internet Joke...   \n",
       "3   Trump Is So Obsessed He Even Has Obama’s Name...   \n",
       "4   Pope Francis Just Called Out Donald Trump Dur...   \n",
       "5   Racist Alabama Cops Brutalize Black Boy While...   \n",
       "6   Fresh Off The Golf Course, Trump Lashes Out A...   \n",
       "7   Trump Said Some INSANELY Racist Stuff Inside ...   \n",
       "8   Former CIA Director Slams Trump Over UN Bully...   \n",
       "9   WATCH: Brand-New Pro-Trump Ad Features So Muc...   \n",
       "\n",
       "                                                text subject  \\\n",
       "0  Donald Trump just couldn t wish all Americans ...    News   \n",
       "1  House Intelligence Committee Chairman Devin Nu...    News   \n",
       "2  On Friday, it was revealed that former Milwauk...    News   \n",
       "3  On Christmas day, Donald Trump announced that ...    News   \n",
       "4  Pope Francis used his annual Christmas Day mes...    News   \n",
       "5  The number of cases of cops brutalizing and ki...    News   \n",
       "6  Donald Trump spent a good portion of his day a...    News   \n",
       "7  In the wake of yet another court decision that...    News   \n",
       "8  Many people have raised the alarm regarding th...    News   \n",
       "9  Just when you might have thought we d get a br...    News   \n",
       "\n",
       "                date  class  \n",
       "0  December 31, 2017      0  \n",
       "1  December 31, 2017      0  \n",
       "2  December 30, 2017      0  \n",
       "3  December 29, 2017      0  \n",
       "4  December 25, 2017      0  \n",
       "5  December 25, 2017      0  \n",
       "6  December 23, 2017      0  \n",
       "7  December 23, 2017      0  \n",
       "8  December 22, 2017      0  \n",
       "9  December 21, 2017      0  "
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Merging True and Fake DataFrames\n",
    "df_merge = pd.concat([df_fake, df_true], axis =0 )\n",
    "df_merge.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "030806a5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['title', 'text', 'subject', 'date', 'class'], dtype='object')"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_merge.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "d5ba73fc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['class'], dtype='object')"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Finding columns Numerical Features\n",
    "numerical_cols = df_merge.select_dtypes(include=['int64', 'float64']).columns\n",
    "numerical_cols"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "624c3ef5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAskAAAIhCAYAAAC8IicCAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8qNh9FAAAACXBIWXMAAA9hAAAPYQGoP6dpAABRU0lEQVR4nO3deXxU5d3///eZmcxkIRkSQjZZRAUEwQ0qmxaQRZGliv250FKwuH2pCxW+Vut9V+ytUrWiVevy9aa4gGKrYm2pCIobCi5oVBYBlVUSQiD7MpPMXL8/kozMCVtCkpkkr+fjMQ+Zc64553NyAN9cc53rsowxRgAAAABCHJEuAAAAAIg2hGQAAADAhpAMAAAA2BCSAQAAABtCMgAAAGBDSAYAAABsCMkAAACADSEZAAAAsCEkAwAAADaEZABt1jPPPCPLskKv2NhYZWRkaOTIkZo3b57y8vLqfWbu3LmyLKtB5ykvL9fcuXP17rvvNuhzhzrXiSeeqAkTJjToOEfzwgsv6OGHHz7kPsuyNHfu3CY9X1N7++23NXDgQCUkJMiyLL322mvHdbzWcM0AIs8V6QIAoLktXLhQp556qqqqqpSXl6fVq1frvvvu05///Ge99NJLGj16dKjt1VdfrQsvvLBBxy8vL9ddd90lSRoxYsQxf64x52qMF154QevXr9esWbPq7VuzZo26dOnS7DU0ljFGl112mXr16qXXX39dCQkJ6t27d6TLAtAOEJIBtHn9+vXTwIEDQ+8vvfRS/fa3v9W5556ryZMna+vWrUpPT5ckdenSpdlDY3l5ueLj41vkXEczePDgiJ7/aPbs2aMDBw7okksu0ahRoyJdDoB2hOEWANqlbt266cEHH1RJSYmeeuqp0PZDDYFYtWqVRowYoU6dOikuLk7dunXTpZdeqvLycm3fvl2dO3eWJN11112hoR3Tp08PO97nn3+un//850pOTtbJJ5982HPVWbp0qU4//XTFxsbqpJNO0iOPPBK2v24oyfbt28O2v/vuu7IsKzT0Y8SIEVq2bJl27NgRNvSkzqGGHqxfv14/+9nPlJycrNjYWJ155pl69tlnD3meF198UXfccYeysrKUlJSk0aNHa/PmzYf/wR9k9erVGjVqlBITExUfH6+hQ4dq2bJlof1z584N/SPid7/7nSzL0oknnnjEYxYWFmr27Nk66aST5PF4lJaWposuukjffPPNYT+zb98+zZw5U3379lWHDh2Ulpam888/Xx988EG9tk888YTOOOMMdejQQYmJiTr11FP1+9//PrS/vLxcc+bMUY8ePRQbG6uUlBQNHDhQL7744jH9TABED3qSAbRbF110kZxOp95///3Dttm+fbvGjx+v8847T3/729/UsWNH/fDDD1q+fLn8fr8yMzO1fPlyXXjhhZoxY4auvvpqSQoF5zqTJ0/WFVdcoeuvv15lZWVHrCs7O1uzZs3S3LlzlZGRocWLF+vmm2+W3+/XnDlzGnSNjz/+uK699lp99913Wrp06VHbb968WUOHDlVaWpoeeeQRderUSYsWLdL06dO1d+9e3XrrrWHtf//732vYsGH63//9XxUXF+t3v/udJk6cqE2bNsnpdB72PO+9957GjBmj008/XQsWLJDH49Hjjz+uiRMn6sUXX9Tll1+uq6++WmeccYYmT56sG2+8UVOmTJHH4znsMUtKSnTuuedq+/bt+t3vfqdBgwaptLRU77//vnJycnTqqace8nMHDhyQJN15553KyMhQaWmpli5dqhEjRujtt98ODaFZsmSJZs6cqRtvvFF//vOf5XA49O2332rjxo2hY91yyy16/vnndffdd+uss85SWVmZ1q9fr/379x/1Zw8gyhgAaKMWLlxoJJlPP/30sG3S09NNnz59Qu/vvPNOc/BfjS+//LKRZLKzsw97jH379hlJ5s4776y3r+54f/jDHw6772Ddu3c3lmXVO9+YMWNMUlKSKSsrC7u2bdu2hbV75513jCTzzjvvhLaNHz/edO/e/ZC12+u+4oorjMfjMTt37gxrN27cOBMfH28KCwvDznPRRReFtfv73/9uJJk1a9Yc8nx1Bg8ebNLS0kxJSUloW3V1tenXr5/p0qWLCQaDxhhjtm3bZiSZBx544IjHM8aYP/7xj0aSWbly5RHbHe5eHVxHVVWVGTVqlLnkkktC22+44QbTsWPHIx67X79+5uKLLz5qrQCiH8MtALRrxpgj7j/zzDPldrt17bXX6tlnn9X333/fqPNceumlx9z2tNNO0xlnnBG2bcqUKSouLtbnn3/eqPMfq1WrVmnUqFHq2rVr2Pbp06ervLxca9asCds+adKksPenn366JGnHjh2HPUdZWZk+/vhj/fznP1eHDh1C251Op6ZOnardu3cf85CNg73xxhvq1atX2IOYx+rJJ5/U2WefrdjYWLlcLsXExOjtt9/Wpk2bQm3OOeccFRYW6sorr9Q///lP5efn1zvOOeecozfeeEO33Xab3n33XVVUVDS4FgDRgZAMoN0qKyvT/v37lZWVddg2J598st566y2lpaXpN7/5jU4++WSdfPLJ+stf/tKgc2VmZh5z24yMjMNua+6v7ffv33/IWut+Rvbzd+rUKex93XCII4XDgoICGWMadJ5jsW/fvkY9CDl//nz9n//zfzRo0CC98sorWrt2rT799FNdeOGFYdcxdepU/e1vf9OOHTt06aWXKi0tTYMGDdLKlStDbR555BH97ne/02uvvaaRI0cqJSVFF198sbZu3drgugBEFiEZQLu1bNkyBQKBo07bdt555+lf//qXioqKtHbtWg0ZMkSzZs3SkiVLjvlcDZl7OTc397Db6kJpbGysJMnn84W1O1TvZkN06tRJOTk59bbv2bNHkpSamnpcx5ek5ORkORyOJj9P586dtXv37gZ/btGiRRoxYoSeeOIJjR8/XoMGDdLAgQNVUlJSr+1VV12ljz76SEVFRVq2bJmMMZowYUKo5zwhIUF33XWXvvnmG+Xm5uqJJ57Q2rVrNXHixAbXBSCyCMkA2qWdO3dqzpw58nq9uu66647pM06nU4MGDdJf//pXSQoNfTiW3tOG2LBhg7788suwbS+88IISExN19tlnS1JoloevvvoqrN3rr79e73gej+eYaxs1apRWrVoVCqt1nnvuOcXHxzfJlHEJCQkaNGiQXn311bC6gsGgFi1apC5duqhXr14NPu64ceO0ZcsWrVq1qkGfsyyr3gOBX331Vb2hJQdLSEjQuHHjdMcdd8jv92vDhg312qSnp2v69Om68sortXnzZpWXlzeoLgCRxewWANq89evXq7q6WtXV1crLy9MHH3yghQsXyul0aunSpfVmojjYk08+qVWrVmn8+PHq1q2bKisr9be//U2SQmNfExMT1b17d/3zn//UqFGjlJKSotTU1KNOV3Y4WVlZmjRpkubOnavMzEwtWrRIK1eu1H333af4+HhJ0k9+8hP17t1bc+bMUXV1tZKTk7V06VKtXr263vH69++vV199VU888YQGDBggh8MRNm/0we688079+9//1siRI/WHP/xBKSkpWrx4sZYtW6b7779fXq+3Uddk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JemCCy6Qz+fTunXrDlubz+dTcXFx2AsAAACtX1SH5HHjxmnx4sVatWqVHnzwQX366ac6//zz5fP5JEm5ublyu91KTk4O+1x6erpyc3NDbdLS0uodOy0tLaxNenp62P7k5GS53e5Qm0OZN29eaJyz1+tV165dj+t6AQAAEB0iOtziaC6//PLQr/v166eBAweqe/fuWrZsmSZPnnzYzxljZFlW6P3Bvz6eNna33367brnlltD74uJigjIAAEAbENU9yXaZmZnq3r27tm7dKknKyMiQ3+9XQUFBWLu8vLxQz3BGRob27t1b71j79u0La2PvMS4oKFBVVVW9HuaDeTweJSUlhb0AAADQ+rWqkLx//37t2rVLmZmZkqQBAwYoJiZGK1euDLXJycnR+vXrNXToUEnSkCFDVFRUpE8++STU5uOPP1ZRUVFYm/Xr1ysnJyfUZsWKFfJ4PBowYEBLXBoAAACiSESHW5SWlurbb78Nvd+2bZuys7OVkpKilJQUzZ07V5deeqkyMzO1fft2/f73v1dqaqouueQSSZLX69WMGTM0e/ZsderUSSkpKZozZ4769+8fmu2iT58+uvDCC3XNNdfoqaeekiRde+21mjBhgnr37i1JGjt2rPr27aupU6fqgQce0IEDBzRnzhxdc8019A4DAAC0QxENyZ999plGjhwZel83vnfatGl64okn9PXXX+u5555TYWGhMjMzNXLkSL300ktKTEwMfeahhx6Sy+XSZZddpoqKCo0aNUrPPPOMnE5nqM3ixYt10003hWbBmDRpUtjczE6nU8uWLdPMmTM1bNgwxcXFacqUKfrzn//c3D8CAAAARKGIhuQRI0bIGHPY/W+++eZRjxEbG6tHH31Ujz766GHbpKSkaNGiRUc8Trdu3fTvf//7qOcDAABA29eqxiQDAAAALYGQDAAAANgQkgEAAAAbQjIAAABgQ0gGAAAAbAjJAAAAgA0hGQAAALAhJAMAAAA2hGQAAADAhpAMAAAA2DQqJG/btq2p6wAAAACiRqNC8imnnKKRI0dq0aJFqqysbOqaAAAAgIhqVEj+8ssvddZZZ2n27NnKyMjQddddp08++aSpawMAAAAiolEhuV+/fpo/f75++OEHLVy4ULm5uTr33HN12mmnaf78+dq3b19T1wkAAAC0mON6cM/lcumSSy7R3//+d91333367rvvNGfOHHXp0kW/+tWvlJOT01R1AgAAAC3muELyZ599ppkzZyozM1Pz58/XnDlz9N1332nVqlX64Ycf9LOf/ayp6gQAAABajKsxH5o/f74WLlyozZs366KLLtJzzz2niy66SA5HTebu0aOHnnrqKZ166qlNWiwAAADQEhoVkp944gn9+te/1lVXXaWMjIxDtunWrZsWLFhwXMUBAAAAkdCokLx169ajtnG73Zo2bVpjDg8AAABEVKPGJC9cuFD/+Mc/6m3/xz/+oWefffa4iwIAAAAiqVEh+U9/+pNSU1PrbU9LS9O999573EUBAAAAkdSokLxjxw716NGj3vbu3btr586dx10UAAAAEEmNCslpaWn66quv6m3/8ssv1alTp+MuCgAAAIikRoXkK664QjfddJPeeecdBQIBBQIBrVq1SjfffLOuuOKKpq4RAAAAaFGNmt3i7rvv1o4dOzRq1Ci5XDWHCAaD+tWvfsWYZAAAALR6jQrJbrdbL730kv7nf/5HX375peLi4tS/f3917969qesDAAAAWlyjQnKdXr16qVevXk1VCwAAABAVGhWSA4GAnnnmGb399tvKy8tTMBgM279q1aomKQ4AAACIhEaF5JtvvlnPPPOMxo8fr379+smyrKauCwAAAIiYRoXkJUuW6O9//7suuuiipq4HAAAAiLhGTQHndrt1yimnNHUtAAAAQFRoVEiePXu2/vKXv8gY09T1AAAAABHXqOEWq1ev1jvvvKM33nhDp512mmJiYsL2v/rqq01SHAAAABAJjQrJHTt21CWXXNLUtQAAAABRoVEheeHChU1dBwAAABA1GjUmWZKqq6v11ltv6amnnlJJSYkkac+ePSotLW2y4gAAAIBIaFRP8o4dO3ThhRdq586d8vl8GjNmjBITE3X//fersrJSTz75ZFPXCQAAALSYRvUk33zzzRo4cKAKCgoUFxcX2n7JJZfo7bffbrLiAAAAgEho9OwWH374odxud9j27t2764cffmiSwgAAAIBIaVRPcjAYVCAQqLd99+7dSkxMPO6iAAAAgEhqVEgeM2aMHn744dB7y7JUWlqqO++8k6WqAQAA0Oo1arjFQw89pJEjR6pv376qrKzUlClTtHXrVqWmpurFF19s6hoBAACAFtWokJyVlaXs7Gy9+OKL+vzzzxUMBjVjxgz94he/CHuQDwAAAGiN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Xbtq165dSkpKavbzoelxD1s37l/rxz1s/biHrVsk7p8xRiUlJcrKyjpiO0JyrdTUVDmdTuXm5oZtz8vLU3p6+iE/4/F45PF4wrZ17NixuUo8rKSkJP5iaOW4h60b96/14x62ftzD1q2l75/X6z1qGx7ZrOV2uzVgwACtXLkybPvKlSs1dOjQCFUFAACASKAn+SC33HKLpk6dqoEDB2rIkCH6f//v/2nnzp26/vrrI10aAAAAWhAh+SCXX3659u/frz/+8Y/KyclRv3799J///Efdu3ePdGmH5PF4dOedd9Yb8oHWg3vYunH/Wj/uYevHPWzdovn+WcY05wyWAAAAQOvDmGQAAADAhpAMAAAA2BCSAQAAABtCMgAAAGBDSI5yjz/+uHr06KHY2FgNGDBAH3zwwRHbv/feexowYIBiY2N10kkn6cknn2yhSnEoDbl/r776qsaMGaPOnTsrKSlJQ4YM0ZtvvtmC1eJQGvpnsM6HH34ol8ulM888s3kLxFE19B76fD7dcccd6t69uzwej04++WT97W9/a6FqcSgNvYeLFy/WGWecofj4eGVmZuqqq67S/v37W6haHOz999/XxIkTlZWVJcuy9Nprrx31M1GTZY64aDUiasmSJSYmJsY8/fTTZuPGjebmm282CQkJZseOHYds//3335v4+Hhz8803m40bN5qnn37axMTEmJdffrmFK4cxDb9/N998s7nvvvvMJ598YrZs2WJuv/12ExMTYz7//PMWrhx1GnoP6xQWFpqTTjrJjB071pxxxhktUywOqTH3cNKkSWbQoEFm5cqVZtu2bebjjz82H374YQtWjYM19B5+8MEHxuFwmL/85S/m+++/Nx988IE57bTTzMUXX9zClcMYY/7zn/+YO+64w7zyyitGklm6dOkR20dTliEkR7FzzjnHXH/99WHbTj31VHPbbbcdsv2tt95qTj311LBt1113nRk8eHCz1YjDa+j9O5S+ffuau+66q6lLwzFq7D28/PLLzX/913+ZO++8k5AcYQ29h2+88Ybxer1m//79LVEejkFD7+EDDzxgTjrppLBtjzzyiOnSpUuz1YhjcywhOZqyDMMtopTf79e6des0duzYsO1jx47VRx99dMjPrFmzpl77Cy64QJ999pmqqqqarVbU15j7ZxcMBlVSUqKUlJTmKBFH0dh7uHDhQn333Xe68847m7tEHEVj7uHrr7+ugQMH6v7779cJJ5ygXr16ac6cOaqoqGiJkmHTmHs4dOhQ7d69W//5z39kjNHevXv18ssva/z48S1RMo5TNGUZVtyLUvn5+QoEAkpPTw/bnp6ertzc3EN+Jjc395Dtq6urlZ+fr8zMzGarF+Eac//sHnzwQZWVlemyyy5rjhJxFI25h1u3btVtt92mDz74QC4Xf71GWmPu4ffff6/Vq1crNjZWS5cuVX5+vmbOnKkDBw4wLjkCGnMPhw4dqsWLF+vyyy9XZWWlqqurNWnSJD366KMtUTKOUzRlGXqSo5xlWWHvjTH1th2t/aG2o2U09P7VefHFFzV37ly99NJLSktLa67ycAyO9R4GAgFNmTJFd911l3r16tVS5eEYNOTPYTAYlGVZWrx4sc455xxddNFFmj9/vp555hl6kyOoIfdw48aNuummm/SHP/xB69at0/Lly7Vt2zZdf/31LVEqmkC0ZBm6OqJUamqqnE5nvX8p5+Xl1fsXVp2MjIxDtne5XOrUqVOz1Yr6GnP/6rz00kuaMWOG/vGPf2j06NHNWSaOoKH3sKSkRJ999pm++OIL3XDDDZJqApcxRi6XSytWrND555/fIrWjRmP+HGZmZuqEE06Q1+sNbevTp4+MMdq9e7d69uzZrDUjXGPu4bx58zRs2DD93//7fyVJp59+uhISEnTeeefp7rvv5lvVKBdNWYae5Cjldrs1YMAArVy5Mmz7ypUrNXTo0EN+ZsiQIfXar1ixQgMHDlRMTEyz1Yr6GnP/pJoe5OnTp+uFF15g/FyENfQeJiUl6euvv1Z2dnbodf3116t3797Kzs7WoEGDWqp01GrMn8Nhw4Zpz549Ki0tDW3bsmWLHA6HunTp0qz1or7G3MPy8nI5HOHxxul0SvqxRxLRK6qyTIs/KohjVjftzYIFC8zGjRvNrFmzTEJCgtm+fbsxxpjbbrvNTJ06NdS+btqU3/72t2bjxo1mwYIFTAEXQQ29fy+88IJxuVzmr3/9q8nJyQm9CgsLI3UJ7V5D76Eds1tEXkPvYUlJienSpYv5+c9/bjZs2GDee+8907NnT3P11VdH6hLavYbew4ULFxqXy2Uef/xx891335nVq1ebgQMHmnPOOSdSl9CulZSUmC+++MJ88cUXRpKZP3+++eKLL0JT+EVzliEkR7m//vWvpnv37sbtdpuzzz7bvPfee6F906ZNM8OHDw9r/+6775qzzjrLuN1uc+KJJ5onnniihSvGwRpy/4YPH24k1XtNmzat5QtHSEP/DB6MkBwdGnoPN23aZEaPHm3i4uJMly5dzC233GLKy8tbuGocrKH38JFHHjF9+/Y1cXFxJjMz0/ziF78wu3fvbuGqYYwx77zzzhH/3xbNWcYyhu8eAAAAgIMxJhkAAACwISQDAAAANoRkAAAAwIaQDAAAANgQkgEAAAAbQjIAAABgQ0gGAAAAbAjJAAAAgA0hGQDaqe3bt8uyLGVnZ0e6FACIOoRkAAAAwIaQDAAAANgQkgGgjQsGg7rvvvt0yimnyOPxqFu3brrnnnvqtQsEApoxY4Z69OihuLg49e7dW3/5y1/C2rz77rs655xzlJCQoI4dO2rYsGHasWOHJOnLL7/UyJEjlZiYqKSkJA0YMECfffZZi1wjADQ1V6QLAAA0r9tvv11PP/20HnroIZ177rnKycnRN998U69dMBhUly5d9Pe//12pqan66KOPdO211yozM1OXXXaZqqurdfHFF+uaa67Riy++KL/fr08++USWZUmSfvGLX+iss87SE088IafTqezsbMXExLT05QJAk7CMMSbSRQAAmkdJSYk6d+6sxx57TFdffXXYvu3bt6tHjx764osvdOaZZx7y87/5zW+0d+9evfzyyzpw4IA6deqkd999V8OHD6/XNikpSY8++qimTZvWHJcCAC2K4RYA0IZt2rRJPp9Po0aNOqb2Tz75pAYOHKjOnTurQ4cOevrpp7Vz505JUkpKiqZPn64LLrhAEydO1F/+8hfl5OSEPnvLLbfo6quv1ujRo/WnP/1J3333XbNcEwC0BEIyALRhcXFxx9z273//u37729/q17/+tVasWKHs7GxdddVV8vv9oTYLFy7UmjVrNHToUL300kvq1auX1q5dK0maO3euNmzYoPHjx2vVqlXq27evli5d2uTXBAAtgeEWANCGVVZWKiUlRY888shRh1vceOON2rhxo95+++1Qm9GjRys/P/+wcykPGTJEP/nJT/TII4/U23fllVeqrKxMr7/+epNeEwC0BHqSAaANi42N1e9+9zvdeuuteu655/Tdd99p7dq1WrBgQb22p5xyij777DO9+eab2rJli/77v/9bn376aWj/tm3bdPvtt2vNmjXasWOHVqxYoS1btqhPnz6qqKjQDTfcoHfffVc7duzQhx9+qE8//VR9+vRpycsFgCbD7BYA0Mb993//t1wul/7whz9oz549yszM1PXXX1+v3fXXX6/s7GxdfvnlsixLV155pWbOnKk33nhDkhQfH69vvvlGzz77rPbv36/MzEzdcMMNuu6661RdXa39+/frV7/6lfbu3avU1FRNnjxZd911V0tfLgA0CYZbAAAAADYMtwAAAABsCMkAAACADSEZAAAAsCEkAwAAADaEZAAAAMCGkAwAAADYEJIBAAAAG0IyAAAAYENIBgAAAGwIyQAAAIANIRkAAACw+f8BS8TtyiJaJy0AAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 800x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Distribution of Numerical Features\n",
    "numerical_cols = df_merge.select_dtypes(include=['int64', 'float64']).columns\n",
    "for col in numerical_cols:\n",
    "    plt.figure(figsize=(8, 6))\n",
    "    sns.histplot(df_merge[col], kde=True, color='skyblue')\n",
    "    plt.title(f'Distribution of {col}')\n",
    "    plt.xlabel(col)\n",
    "    plt.ylabel('Frequency')\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "13bf783b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "text     0\n",
       "class    0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Removing columns which are not required\n",
    "df = df_merge.drop([\"title\", \"subject\",\"date\"], axis = 1)\n",
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "3d6cb21a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>text</th>\n",
       "      <th>class</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>657</th>\n",
       "      <td>(Reuters) - U.S. Senator Rand Paul of Kentucky...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13283</th>\n",
       "      <td>HANOVER, Germany (Reuters) - Members of the an...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>18494</th>\n",
       "      <td>DUBAI (Reuters) - Five Bahraini policemen were...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7325</th>\n",
       "      <td>LOS ANGELES/WASHINGTON (Reuters) - Americans’ ...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10075</th>\n",
       "      <td>This is mob rule AGAIN! The best part of this ...</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                    text  class\n",
       "657    (Reuters) - U.S. Senator Rand Paul of Kentucky...      1\n",
       "13283  HANOVER, Germany (Reuters) - Members of the an...      1\n",
       "18494  DUBAI (Reuters) - Five Bahraini policemen were...      1\n",
       "7325   LOS ANGELES/WASHINGTON (Reuters) - Americans’ ...      1\n",
       "10075  This is mob rule AGAIN! The best part of this ...      0"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Random Shuffling the dataframe\n",
    "df = df.sample(frac = 1)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "5be63b39",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Resetting index\n",
    "df.reset_index(inplace = True)\n",
    "df.drop([\"index\"], axis = 1, inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "99e014b0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['text', 'class'], dtype='object')"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "26b77fdb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>text</th>\n",
       "      <th>class</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>(Reuters) - U.S. Senator Rand Paul of Kentucky...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>HANOVER, Germany (Reuters) - Members of the an...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>DUBAI (Reuters) - Five Bahraini policemen were...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>LOS ANGELES/WASHINGTON (Reuters) - Americans’ ...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>This is mob rule AGAIN! The best part of this ...</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                text  class\n",
       "0  (Reuters) - U.S. Senator Rand Paul of Kentucky...      1\n",
       "1  HANOVER, Germany (Reuters) - Members of the an...      1\n",
       "2  DUBAI (Reuters) - Five Bahraini policemen were...      1\n",
       "3  LOS ANGELES/WASHINGTON (Reuters) - Americans’ ...      1\n",
       "4  This is mob rule AGAIN! The best part of this ...      0"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "dd7b152e",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\hp\\AppData\\Local\\Temp\\ipykernel_23124\\662795431.py:2: FutureWarning: Dropping of nuisance columns in DataFrame reductions (with 'numeric_only=None') is deprecated; in a future version this will raise TypeError.  Select only valid columns before calling the reduction.\n",
      "  df.skew()\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "class    0.092083\n",
       "dtype: float64"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Data Skewness\n",
    "df.skew()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "19cee38d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>class</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>44878.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>0.477004</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>0.499476</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              class\n",
       "count  44878.000000\n",
       "mean       0.477004\n",
       "std        0.499476\n",
       "min        0.000000\n",
       "25%        0.000000\n",
       "50%        0.000000\n",
       "75%        1.000000\n",
       "max        1.000000"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Data Distribution\n",
    "df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "89482fb8",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Creating a function to process the texts\n",
    "def wordopt(text):\n",
    "    text = text.lower()\n",
    "    text = re.sub('\\[.*?\\]', '', text)\n",
    "    text = re.sub(\"\\\\W\",\" \",text) \n",
    "    text = re.sub('https?://\\S+|www\\.\\S+', '', text)\n",
    "    text = re.sub('<.*?>+', '', text)\n",
    "    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)\n",
    "    text = re.sub('\\n', '', text)\n",
    "    text = re.sub('\\w*\\d\\w*', '', text)    \n",
    "    return text"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "b1949061",
   "metadata": {},
   "outputs": [],
   "source": [
    "df[\"text\"] = df[\"text\"].apply(wordopt)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "2775580e",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Defining dependent and independent variables\n",
    "x = df[\"text\"]\n",
    "y = df[\"class\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "30d20e1f",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Splitting Training and Testing\n",
    "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "ea2d002c",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Convert text to vectors\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "vectorization = TfidfVectorizer()\n",
    "xv_train = vectorization.fit_transform(x_train)\n",
    "xv_test = vectorization.transform(x_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "0fda2301",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LogisticRegression()"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Logistic Regression -- Model 1\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "\n",
    "LR = LogisticRegression()\n",
    "LR.fit(xv_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "dc2405c5",
   "metadata": {},
   "outputs": [],
   "source": [
    "pred_lr=LR.predict(xv_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "0bf7ecab",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9852941176470589"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "LR.score(xv_test, y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "0c84f011",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.99      0.98      0.99      5765\n",
      "           1       0.98      0.99      0.98      5455\n",
      "\n",
      "    accuracy                           0.99     11220\n",
      "   macro avg       0.99      0.99      0.99     11220\n",
      "weighted avg       0.99      0.99      0.99     11220\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(classification_report(y_test, pred_lr))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "5b7b19da",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LinearRegression()"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Linear Regression --Model 2\n",
    "from sklearn.linear_model import LinearRegression\n",
    "lin_reg = LinearRegression()\n",
    "lin_reg.fit(xv_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "dbf74311",
   "metadata": {},
   "outputs": [],
   "source": [
    "lin_reg_pred = lin_reg.predict(xv_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "b4789c96",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.7441612796792743"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "lin_reg.score(xv_test, y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "216f5092",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "DecisionTreeClassifier()"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Decision Tree Classification --Model 3\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "\n",
    "DT = DecisionTreeClassifier()\n",
    "DT.fit(xv_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "33a56b59",
   "metadata": {},
   "outputs": [],
   "source": [
    "pred_dt = DT.predict(xv_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "798ce301",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9952762923351158"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "DT.score(xv_test, y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "45aa7e6d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.99      1.00      1.00      5765\n",
      "           1       1.00      0.99      1.00      5455\n",
      "\n",
      "    accuracy                           1.00     11220\n",
      "   macro avg       1.00      1.00      1.00     11220\n",
      "weighted avg       1.00      1.00      1.00     11220\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(classification_report(y_test, pred_dt))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "76480ccd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RandomForestClassifier(random_state=0)"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Random Forest Classifier --Model 4\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "\n",
    "RFC = RandomForestClassifier(random_state=0)\n",
    "RFC.fit(xv_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "0d61d06b",
   "metadata": {},
   "outputs": [],
   "source": [
    "pred_rfc = RFC.predict(xv_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "2fe37d92",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9894830659536542"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "RFC.score(xv_test, y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "2e9f8ff8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.99      0.99      0.99      5765\n",
      "           1       0.99      0.99      0.99      5455\n",
      "\n",
      "    accuracy                           0.99     11220\n",
      "   macro avg       0.99      0.99      0.99     11220\n",
      "weighted avg       0.99      0.99      0.99     11220\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(classification_report(y_test, pred_rfc))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "89054a32",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Model Testing\n",
    "def output_lable(n):\n",
    "    if n == 0:\n",
    "        return \"Fake News\"\n",
    "    elif n == 1:\n",
    "        return \"Not A Fake News\"\n",
    "    \n",
    "def manual_testing(news):\n",
    "    testing_news = {\"text\":[news]}\n",
    "    new_def_test = pd.DataFrame(testing_news)\n",
    "    new_def_test[\"text\"] = new_def_test[\"text\"].apply(wordopt) \n",
    "    new_x_test = new_def_test[\"text\"]\n",
    "    new_xv_test = vectorization.transform(new_x_test)\n",
    "    pred_LR = LR.predict(new_xv_test)\n",
    "    pred_DT = DT.predict(new_xv_test)\n",
    "    pred_RFC = RFC.predict(new_xv_test)\n",
    "\n",
    "    return print(\"\\n\\nLR Prediction: {} \\nDT Prediction: {}  \\nRFC Prediction: {}\".format(output_lable(pred_LR[0]),                                                                                                       output_lable(pred_DT[0]), \n",
    "                                                                                                              output_lable(pred_DT[0]), \n",
    "                                                                                                              output_lable(pred_RFC[0])))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "ef9a23e8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Vic Bishop Waking TimesOur reality is carefully constructed by powerful corporate, political and special interest sources in order to covertly sway public opinion. Blatant lies are often televised regarding terrorism, food, war, health, etc. They are fashioned to sway public opinion and condition viewers to accept what have become destructive societal norms.The practice of manipulating and controlling public opinion with distorted media messages has become so common that there is a whole industry formed around this. The entire role of this brainwashing industry is to figure out how to spin information to journalists, similar to the lobbying of government. It is never really clear just how much truth the journalists receive because the news industry has become complacent. The messages that it presents are shaped by corporate powers who often spend millions on advertising with the six conglomerates that own 90% of the media:General Electric (GE), News-Corp, Disney, Viacom, Time Warner, and CBS. Yet, these corporations function under many different brands, such as FOX, ABC, CNN, Comcast, Wall Street Journal, etc, giving people the perception of choice   As Tavistock s researchers showed, it was important that the victims of mass brainwashing not be aware that their environment was being controlled; there should thus be a vast number of sources for information, whose messages could be varied slightly, so as to mask the sense of external control. ~ Specialist of mass brainwashing, L. WolfeNew Brainwashing Tactic Called AstroturfWith alternative media on the rise, the propaganda machine continues to expand. Below is a video of Sharyl Attkisson, investigative reporter with CBS, during which she explains how  astroturf,  or fake grassroots movements, are used to spin information not only to influence journalists but to sway public opinion. Astroturf is a perversion of grassroots. Astroturf is when political, corporate or other special interests disguise themselves and publish blogs, start facebook and twitter accounts, publish ads, letters to the editor, or simply post comments online, to try to fool you into thinking an independent or grassroots movement is speaking. ~ Sharyl Attkisson, Investigative ReporterHow do you separate fact from fiction? Sharyl Attkisson finishes her talk with some insights on how to identify signs of propaganda and astroturfing  These methods are used to give people the impression that there is widespread support for an agenda, when, in reality, one may not exist. Astroturf tactics are also used to discredit or criticize those that disagree with certain agendas, using stereotypical names such as conspiracy theorist or quack. When in fact when someone dares to reveal the truth or questions the  official  story, it should spark a deeper curiosity and encourage further scrutiny of the information.This article (Journalist Reveals Tactics Brainwashing Industry Uses to Manipulate the Public) was originally created and published by Waking Times and is published here under a Creative Commons license with attribution to Vic Bishop and WakingTimes.com. It may be re-posted freely with proper attribution, author bio, and this copyright statement. READ MORE MSM PROPAGANDA NEWS AT: 21st Century Wire MSM Watch Files\n",
      "\n",
      "\n",
      "LR Prediction: Fake News \n",
      "DT Prediction: Fake News  \n",
      "RFC Prediction: Fake News\n"
     ]
    }
   ],
   "source": [
    "news = str(input())\n",
    "manual_testing(news)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "416ad4bf",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "SAO PAULO (Reuters) - Cesar Mata Pires, the owner and co-founder of Brazilian engineering conglomerate OAS SA, one of the largest companies involved in Brazil s corruption scandal, died on Tuesday. He was 68. Mata Pires died of a heart attack while taking a morning walk in an upscale district of S o Paulo, where OAS is based, a person with direct knowledge of the matter said. Efforts to contact his family were unsuccessful. OAS declined to comment. The son of a wealthy cattle rancher in the northeastern state of Bahia, Mata Pires  links to politicians were central to the expansion of OAS, which became Brazil s No. 4 builder earlier this decade, people familiar with his career told Reuters last year. His big break came when he befriended Antonio Carlos Magalh es, a popular politician who was Bahia governor several times, and eventually married his daughter Tereza. Brazilians joked that OAS stood for  Obras Arranjadas pelo Sogro  - or  Work Arranged by the Father-In-Law.   After years of steady growth triggered by a flurry of massive government contracts, OAS was ensnared in Operation Car Wash which unearthed an illegal contracting ring between state firms and builders. The ensuing scandal helped topple former Brazilian President Dilma Rousseff last year. Trained as an engineer, Mata Pires founded OAS with two colleagues in 1976 to do sub-contracting work for larger rival Odebrecht SA - the biggest of the builders involved in the probe.  Before the scandal, Forbes magazine estimated Mata Pires  fortune at $1.6 billion. He dropped off the magazine s billionaire list in 2015, months after OAS sought bankruptcy protection after the Car Wash scandal. While Mata Pires was never accused of wrongdoing in the investigations, creditors demanded he and his family stay away from the builder s day-to-day operations, people directly involved in the negotiations told Reuters at the time. He is survived by his wife and his two sons.\n",
      "\n",
      "\n",
      "LR Prediction: Not A Fake News \n",
      "DT Prediction: Not A Fake News  \n",
      "RFC Prediction: Not A Fake News\n"
     ]
    }
   ],
   "source": [
    "news = str(input())\n",
    "manual_testing(news)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
