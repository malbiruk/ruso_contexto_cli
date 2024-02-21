import random
import re

import numpy as np
import pandas as pd
import pymystem3
import stanza
# import typer
from navec import Navec
from numpy.linalg import norm
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.prompt import Prompt
from rich.text import Text
from rich.traceback import install

# %%

# app = typer.Typer(rich_markup_mode='rich', add_completion=False)
mystem = pymystem3.Mystem()

palette = {
    'green': 'color(2)',
    'yellow': 'color(3)',
    'red': 'color(1)',
}

navec = Navec.load('navec_hudlit_v1_12B_500K_300d_100q.tar')

with open('good_words.txt', encoding='utf-8') as file_:
    good_words = file_.read().split()

console = Console()


def create_nouns():
    with open('model.txt', encoding='utf-8') as f:
        words_rusvec_all = [i.split(' ', 1)[0] for i in f]

    words_rusvec = []
    for word in words_rusvec_all:
        try:
            if word.split('_', 1)[1] == 'NOUN':
                words_rusvec.append(word.split('_', 1)[0])
        except IndexError:
            continue

    words_rusvec = {mystem.lemmatize(word)[0] for word in words_rusvec
                    if re.fullmatch(r'[а-я]+', word) is not None and
                    not mystem.lemmatize(word)[0].endswith((
                        'сь', 'ся', 'ать', 'ять', 'еть', 'уть', 'оть', 'ыть',
                        'ти', 'чь', 'сть', 'зть', 'ая', 'ый', 'ий', 'ой',
                        'ей', 'ое', 'ее', 'ье', 'ые', 'ие'))}

    print(f'{len(words_rusvec)} words in rusvec')

    words_navec = set(navec.vocab.words[:-2])
    new_nouns = words_rusvec & words_navec

    print(f'{len(new_nouns)} words in rusvec&navec')

    # remove named entities
    nlp = stanza.Pipeline(lang='ru', processors='tokenize,ner')
    new_nouns2 = []

    for i in track(new_nouns, description='Performing NER...'):
        doc_cap = nlp(i.capitalize()).sentences[0]
        ents_cap = doc_cap.ents

        if len(ents_cap) == 1 and ents_cap[0].type == 'MISC':
            new_nouns2.append(i)
        elif len(ents_cap) == 0:
            doc_upper = nlp(i.upper()).sentences[0]
            ents_upper = doc_upper.ents
            if len(ents_upper) == 0 or (len(ents_upper) == 1 and
                                        ents_upper[0].start_char != 0
                                        and ents_upper[0].end_char != len(i)):
                new_nouns2.append(i)

    new_nouns = set(new_nouns2)

    print(f'{len(new_nouns)} words after NER')

    with open('russian_nouns.txt', encoding='utf-8') as f:
        rus_nouns = {mystem.lemmatize(word)[0] for word in f.read().split('\n')
                     if re.fullmatch(r'[а-я]+', mystem.lemmatize(word)[0])
                     is not None}

    bad_words = {'ага', 'больно', 'больше', 'бы', 'бывало', 'быть', 'вдвоём',
                 'вероятно', 'вместе', 'вниз', 'во', 'вовремя', 'возле',
                 'впятером', 'вроде', 'вряд', 'все', 'всегда', 'вскоре',
                 'всё', 'второй', 'втроем', 'вчетвером', 'вшестером', 'где',
                 'достаточно', 'другой', 'единственный', 'есть', 'еще', 'жаль',
                 'затем', 'зачем',  'здесь', 'извините', 'иногда', 'каждый',
                 'когда', 'кому', 'кроме', 'кто', 'куда', 'ладно', 'легко',
                 'лучше', 'мало', 'между', 'меньше', 'много', 'можно', 'на',
                 'некоторые', 'некто', 'некуда', 'нельзя', 'нет', 'ни',
                 'ниоткуда', 'ничего', 'ничто', 'ничуть', 'нужно', 'один',
                 'опять', 'очень', 'первый', 'перед', 'по', 'под',
                 'после', 'потом', 'потому', 'почему', 'почти', 'поэтому',
                 'равно', 'раз', 'раньше', 'редко', 'сам', 'самый', 'свой',
                 'совсем', 'спасибо', 'сразу', 'так', 'также', 'таки', 'такой',
                 'тогда', 'только', 'том', 'точно', 'третий', 'ты', 'хорошо',
                 'чего', 'чему', 'что', 'чтобы', 'чуть', 'это', 'я',
                 'вдруг', 'везде', 'вокруг', 'вокругу', 'всюду', 'да', 'даже',
                 'до', 'за', 'завтра', 'кажется', 'как', 'аркадий', 'над',
                 'никак', 'никакой', 'никто', 'он', 'она', 'они', 'оно',
                 'пожалуй', 'пожалуйста', 'пока',  'просто', 'против',
                 'сегодня', 'снова', 'там', 'то', 'хоть', 'часто', 'надо',
                 }

    new_nouns = (new_nouns | rus_nouns) & words_navec - bad_words

    print(f'{len(new_nouns)} words finally (+ rus_nonuns)')

    with open('nouns.txt', 'w', encoding='utf-8') as f:
        f.write(' '.join(new_nouns))

# %%

# def create_nouns():
#     df = pd.DataFrame(zip(navec.vocab.words, navec.vocab.counts),
# columns=['word', 'count'])
#
#     # remove most often words
    # all_words = df.sort_values(
    #     'count',
    #     ascending=False).reset_index(drop=True)['word'].to_list()[125:-2]
#
#     # remove non-cyrillic symbols and short words
#     all_words = [i for i in all_words if len(i) > 2
# and re.fullmatch(r'[а-яё]+', i)
#  is not None]
#
#     # collect only nouns in lemmantized form
#     new_nouns = []
#     for i in all_words:
#         try:
#             if (mystem.analyze(i)[0]['analysis'][0]['gr'].startswith('S')
#                     and len(mystem.lemmatize(i)[0]) > 2):
#                 new_nouns.append(mystem.lemmatize(i)[0])
#         except IndexError:
#             continue
#     new_nouns = list(dict.fromkeys(new_nouns))
#
#     # remove named entities
#     nlp = stanza.Pipeline(lang='ru', processors='tokenize,ner')
#     new_nouns = [i for i in new_nouns
# if len(nlp(i.capitalize()).sentences[0].ents) == 0]
#
#     with open('nouns_new.txt', 'w', encoding='utf-8') as f:
#         f.write(' '.join(new_nouns))


def create_good_words():
    good_words_ = [mystem.lemmatize(i)[0]
                   for i in pd.read_csv('good_words.tsv',
                   sep='\t').iloc[:, 0].to_list()]
    with open('good_words.txt', 'w', encoding='utf-8') as f:
        f.write(' '.join(good_words_))


def choose_word():
    secret_word = random.choice(good_words)
    while not secret_word in navec.vocab.words:
        secret_word = random.choice(good_words)
    return secret_word


def get_cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))


def create_ratings(secret_word):
    data = []

    with open('nouns.txt', encoding='utf-8') as f:
        nouns = f.read().split()

    nouns.extend(good_words)
    nouns = list(dict.fromkeys(nouns))

    for word in nouns:
        try:
            data.append((
                word, get_cosine_similarity(navec[word], navec[secret_word])))
        except KeyError:
            continue

    words_scores = pd.DataFrame.from_records(data, columns=['word', 'cos_sim'])
    words_scores = (words_scores
                    .sort_values(['cos_sim'], ascending=False)
                    .reset_index(drop=True))
    words_scores['rating'] = words_scores.index + 1

    return words_scores


def cbar_width_by_rating(rating, max_rating, full_width=50):
    if rating <= 500:
        return round(full_width / 2
                     + (full_width - full_width / 2)
                     / 500 * (500 - rating))
    if rating <= 1500:
        return round(full_width / 8
                     + (full_width / 2 - full_width / 8)
                     / 1000 * (1500 - rating))
    return round(full_width / 8 / (max_rating - 1500) * (max_rating - rating))


def color_by_rating(rating):
    if rating <= 300:
        return palette['green']
    if rating <= 1500:
        return palette['yellow']
    return palette['red']


def show_guess(word, rating, max_rating, bold=False):
    width = min(round(0.9 * console.width), 60)
    guess = Text(word + ' ' * (width - 2 - len(word) - len(str(rating)))
                 + str(rating),
                 end='', overflow='crop')

    guess.stylize(f'color(0) on {color_by_rating(rating)}',
                  0, cbar_width_by_rating(rating, max_rating, width))
    if bold:
        guess.stylize('bold')
        console.print(Panel(guess, width=width, padding=0), justify='center')

    else:
        guess.stylize('dim')
        console.print(Panel(guess, width=width, padding=0,
                      border_style='grey46'), justify='center')


def refresh_page(n_guesses, n_tips, commands=True,
                 win=None, guesses=None, secret_word=None):
    console.clear()
    console.rule('[bold]КОНТЕКСТО[/]\n')
    if win is None:
        console.print(f'[bold blue]попыток:[/] {n_guesses}    '
                      f'[bold blue]подсказок:[/] {n_tips}',
                      justify='center')
        if commands:
            console.print('[dim] доступные команды: /мои_попытки, '
                          '/подсказка, /сдаться, /как_играть[/]\n',
                          justify='center')
    else:
        if win:
            console.print('\n[bold]поздравляю![/]\n'
                          f'вы угадали слово [bold magenta]{secret_word}[/]\n'
                          f'за {n_guesses} попыток '
                          f'с использованием {n_tips} подсказок.\n',
                          justify='center')
        else:
            console.print('\n[bold]повезёт в следующий раз![/]\n'
                          'вы сдались, пытаясь угадать слово '
                          f'[bold magenta]{secret_word}[/]\n'
                          f'за {n_guesses} попыток '
                          f'с использованием {n_tips} подсказок.\n',
                          justify='center')
        good_guesses = [i for i in guesses if i[1] <= 300]
        mid_guesses = [i for i in guesses if 300 < i[1] <= 1500]
        bad_guesses = [i for i in guesses if i[1] > 1500]
        console.print(f'[green]{len(good_guesses)}[/]', justify='center')
        console.print(f'[yellow]{len(mid_guesses)}[/]', justify='center')
        console.print(f'[red]{len(bad_guesses)}[/]\n', justify='center')
        console.print('[dim] доступные команды: /мои_попытки, '
                      '/ближайшие_слова, /новая_игра, /выйти[/]\n',
                      justify='center')


def handle_command(guess, guesses, to_refresh, rated_words, n_tips):
    if guess == '/сдаться':
        give_up = Prompt.ask('вы уверены, что хотите сдаться? '
                             r'[bold magenta]\[да/нет][/]')
        if give_up == 'да':
            to_refresh.append('give up')
        else:
            to_refresh.append(False)

    elif guess == '/мои_попытки':
        refresh_page(len(guesses) - n_tips, n_tips)
        show_guesses(guesses, rated_words, n=None)
        to_refresh.append(False)

    elif guess == '/подсказка':
        n_tips += 1
        if len(guesses) == 0:
            tip = rated_words[rated_words['rating'] == 300]
        else:
            best_guess = sorted(guesses, key=lambda tup: tup[1])[0]
            if best_guess[1] == 2:
                c = 1
                while best_guess[1] + c in [i[1] for i in guesses]:
                    c += 1
                tip = rated_words[rated_words['rating'] == best_guess[1] + c]
            else:
                tip = (rated_words[rated_words['rating']
                                   == round(best_guess[1] / 2)])
        guesses.append((tip['word'].to_string(index=False), int(tip['rating'])))
        to_refresh.append(True)

    elif guess == '/как_играть':
        refresh_page(len(guesses) - n_tips, n_tips, False)
        console.print('\nугадайте секретное слово. '
                      'число попыток не ограничено.\n\n'
                      'слова были отсортированы по схожести с секретным словом '
                      'при помощи искусственного интеллекта.\n\n'
                      'после ввода слова вы увидите его рейтинг. '
                      'у секретного слова рейтинг 1.\n\n'
                      'алгоритм проанализировал тысячи текстов. '
                      'он использует контекст, в котором используются слова, '
                      'для вычисления сходства между ними.')
        PyPrompt.ask('\n[dim]введите что угодно, чтобы вернуться[/]')

        to_refresh.append(True)

    return n_tips


def check_guess(guess, guesses, rated_words, after_win=False):
    if guess.startswith('/'):
        if not after_win:
            if not guess in ['/мои_попытки', '/подсказка', '/сдаться',
                             '/как_играть']:
                console.print('[red]я не знаю такой команды :с[/]\n')
                return False
        else:
            if not guess in ['/мои_попытки', '/ближайшие_слова', '/новая_игра',
                             '/выйти']:
                console.print('[red]я не знаю такой команды :с[/]\n')
                # return False

    elif re.fullmatch(r'[а-я]+', mystem.lemmatize(guess)[0]) is None:
        console.print('[red]я не знаю такого слова :с[/]\n')
        return False

    else:
        guess = mystem.lemmatize(guess)[0]

        if not np.any(rated_words['word'].str.fullmatch(guess)):
            console.print('[red]я не знаю такого слова :с[/]\n')
            return False

        if guess in [i[0] for i in guesses]:
            console.print('[red]вы уже угадывали это слово[/]\n')
            return False

    return True


class PyPrompt(Prompt):
    prompt_suffix = Text(': ', style='dim')


def show_guesses(guesses, rated_words, n=10):

    if n is not None:
        n = round((console.height - 15) / 3)
        # print(n)

    if len(guesses) != 0:
        sorted_guesses = sorted(guesses, key=lambda tup: tup[1])[:n]
        for w, r in sorted_guesses:
            bold = (w, r) == guesses[-1]
            show_guess(w, r, len(rated_words), bold=bold)

        console.print()
        show_guess(*guesses[-1], len(rated_words), bold=True)


def handle_command2(guess, guesses, to_refresh, rated_words, n_tips, win,
                    secret_word):
    if guess == '/мои_попытки':
        refresh_page(len(guesses) - n_tips, n_tips, win=win, guesses=guesses,
                     secret_word=secret_word)
        show_guesses(guesses, rated_words, n=None)
        to_refresh.append(False)

    elif guess == '/ближайшие_слова':
        refresh_page(len(guesses) - n_tips, n_tips, win=win, guesses=guesses,
                     secret_word=secret_word)
        for row in rated_words[:500].iterrows():
            show_guess(row[1]['word'], row[1]['rating'], len(rated_words))

        PyPrompt.ask('\n[dim]введите что угодно, чтобы вернуться[/]')
        to_refresh.append(True)

    elif guess == '/выйти':
        out = Prompt.ask('вы уверены, что хотите выйти? '
                         r'[bold magenta]\[да/нет][/]')
        if out == 'да':
            to_refresh.append('out')
        else:
            to_refresh.append(False)

    elif guess == '/новая_игра':
        to_refresh.append('new')


# @ app.command()
def main():
    # create_nouns()
    # exit()
    secret_word = choose_word()
    rated_words = create_ratings(secret_word)

    n_tips = 0

    guesses = []
    to_refresh = [True]

    while True:
        if to_refresh[-1]:
            refresh_page(len(guesses) - n_tips, n_tips)
            show_guesses(guesses, rated_words)

        guess = PyPrompt.ask('\n[dim]введите слово[/]')

        check = check_guess(guess, guesses, rated_words)
        to_refresh.append(check)

        if check:
            if guess.startswith('/'):
                n_tips = handle_command(guess, guesses, to_refresh,
                                        rated_words, n_tips)
                if to_refresh[-1] == 'give up':
                    break
            else:
                guess = mystem.lemmatize(guess)[0]
                guesses.append((
                    guess,
                    int(rated_words[rated_words['word'] == guess]['rating'])))

        to_refresh.pop(0)

        if len(guesses) > 0:
            if guesses[-1][1] == 1:
                to_refresh.append('win')
                break

    win = to_refresh[-1] == 'win'
    to_refresh[-1] = True
    n_guesses = len(guesses) - n_tips
    guesses_old = guesses.copy()
    while True:
        if to_refresh[-1]:
            refresh_page(n_guesses, n_tips, win=win,
                         guesses=guesses_old, secret_word=secret_word)
            show_guesses(guesses, rated_words)

        guess = PyPrompt.ask('\n[dim]введите слово[/]')

        check = check_guess(guess, guesses, rated_words, after_win=True)
        to_refresh.append(check)

        if check:
            if guess.startswith('/'):
                handle_command2(guess, guesses_old, to_refresh,
                                rated_words, n_tips, win,
                                secret_word=secret_word)
                if to_refresh[-1] == 'out':
                    break
                if to_refresh[-1] == 'new':
                    main()
            else:

                guess = mystem.lemmatize(guess)[0]
                guesses.append((
                    guess,
                    int(rated_words[rated_words['word'] == guess]['rating'])))

        to_refresh.pop(0)


if __name__ == '__main__':
    install()  # traceback
    main()
