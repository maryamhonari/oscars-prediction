import csv
import numpy


class Award:
    def __init__(self, name, people, has_won):
        self.name = name
        self.people = people
        self.nominated = True
        self.won = has_won


class Movie:
    def __init__(self, movie_name, year):
        self.name = movie_name
        self.year = year
        self.awards = []
        self.inDataset = False


def compare(x, y):
    if len(x) != len(y):
        return False
    last = 0.0
    for i in range(len(x)):
        if x[i].isalpha() == False:
            if x[i] != y[i]:
                return False
        elif x[i].lower() != y[i].lower():
            return False

    return True

def refine_string(s):
    s = repr(s)
    for j in range(len(s)):
        if s[j] == '\\':
            s = s[0:j]
            break
    return s[1:]

movies = dict()

with open('all_info.txt') as fp:
    for s in fp:
        s = s[0: len(s) - 1]
        list = s.split('|')

        if list[0] not in movies:
            movies[list[0]] = Movie(list[0], list[1])
        has_won = (list[4] == 'won')
        movies[list[0]].awards.append(Award(list[2], list[3], has_won))
#
# for x in movies:
#     print ""
#     print x + "   ================== " + movies[x].year
#     for i in range(len(movies[x].awards)):
#         print 'name = ' + movies[x].awards[i].name + " people = " + movies[x].awards[i].people

print len(movies)

# Handling CSV

csvfile = open('/Users/alimashreghi/Academic/Data_Mining/Project/Dataset/movie_metadata.csv', 'rb')
wb = csv.reader(csvfile, delimiter=',', quotechar='\"')

imdb = []

row_count = 0
column_count = 0
workbook = []

for row in wb:
    workbook.append(row)
    s = row[ord('L') - ord('A')]
    s = refine_string(s)
    row_count += 1
    if row_count > 1:
        column_count = len(row)
        imdb.append(s)

print imdb

# Award categories
awards = ["Actor Leading", "Actor Supporting", "Actress Leading", "Actress Supporting",
            "Animated", "Art Direction", "Cinematography", "Costume Design",
            "Directing", "Film Editing", "Foreign", "Makeup",
            "Music Scoring", "Music Song", "Sound", "Sound Editing",
            "Visual Effects", "Writing", "Best Picture"]

# writing award columns
award_to_index = dict()
for i in range(len(awards)):
    award_to_index[awards[i]] = i

for i in range(2*len(awards) + 1):
    ss = 'Nominated'
    if i%2 == 1:
        ss = 'Won'

    if i < 2*len(awards):
        ss += ' ' + awards[i / 2]
    else:
        ss = 'Num of Awards'
    workbook[0].append(ss)
    for j in range(1, row_count):
        workbook[j].append('0')

# writing winner nominee features
not_found = 0
total_best_nom = 0
for x in movies:
    total_won = 0
    found_idx = -1
    for j in range(len(imdb)):
        if compare(imdb[j], x):
            found_idx = j
            movies[x].inDataset = True
            for k in range(len(movies[x].awards)):
                if movies[x].awards[k].name == 'Best Picture':
                    total_best_nom += 1
                idx = column_count + 2*award_to_index[movies[x].awards[k].name]
                if movies[x].awards[k].nominated:
                    workbook[j + 1][idx] = '1'
                if movies[x].awards[k].won:
                    workbook[j + 1][idx + 1] = '1'
                    total_won += 1
            workbook[j + 1][column_count + 2*len(awards)] = str(total_won)
            break
    # if total_won > 0:
    #     print x
    #     print imdb[found_idx]
    if not movies[x].inDataset:
        if total_won > 0:
            not_found += 1

print 'Not Found Count = ' + str(not_found)
print 'Best Picture Nominated Numbers = ' + str(total_best_nom)

#writing keywords
split_feat = open('/Users/alimashreghi/Academic/Data_Mining/Project/Dataset/split_features_new.csv')
split_feat = csv.reader(split_feat, delimiter=',', quotechar='\"')

cnt = 0
for row in split_feat:
    for i in range(column_count, len(row)):
        workbook[cnt].append(row[i])
    cnt += 1
    #print cnt

with open('Final.csv', 'w') as fp:
    a = csv.writer(fp, delimiter=',', quotechar='\"')
    a.writerows(workbook)
