"""English Streamlit interface backed by the evaluated Russian M8 model."""
import html
import os
import re
import sys
from pathlib import Path

# The app lives in app/; the recommender package is in the project folder one level up.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from recommender.bundle import check_bundle
from recommender.translation import check_translation
from recommender.profile_form import (JOB_TITLES, EXPERIENCE, EDUCATION, RESPONSIBILITIES,
    SKILL_LABELS, skill_options, build_profile, exact_skill_matches, match_caption)

# Set by the EC2 release script; a local run shows "local".
VERSION_LINE = ('Version ' + os.environ.get('APP_VERSION', 'local')
                + (' · commit ' + os.environ['APP_COMMIT'] if os.environ.get('APP_COMMIT') else '')
                + (' · deployed ' + os.environ['APP_DEPLOYED_AT'] if os.environ.get('APP_DEPLOYED_AT') else ''))

st.set_page_config(page_title='Job Recommendation System', page_icon='💼', layout='wide',
                   initial_sidebar_state='collapsed')

STYLE = """
<style>
.block-container{padding-top:1.6rem;max-width:1240px}
.hero{background:linear-gradient(120deg,#0f4c41 0%,#176B5B 50%,#2a9d8f 100%);color:#fff;
      padding:1.7rem 2rem;border-radius:18px;margin-bottom:1.3rem;
      box-shadow:0 6px 18px rgba(23,107,91,.22)}
.hero-title{font-size:2rem;font-weight:700;line-height:1.2;margin:0}
.hero-sub{margin:.45rem 0 0;font-size:1.02rem;opacity:.93;max-width:760px}
.hero-tags{margin-top:.9rem}
.hero-tags span{display:inline-block;margin:0 .45rem .3rem 0;padding:3px 12px;border-radius:999px;
      background:rgba(255,255,255,.16);font-size:.82rem}
.sec{font-size:.78rem;font-weight:700;letter-spacing:.07em;text-transform:uppercase;color:#176B5B;
     margin:.9rem 0 .15rem}
.sec:first-child{margin-top:.1rem}
.job-head{display:flex;gap:.85rem;align-items:flex-start}
.rank{flex:0 0 auto;width:2.1rem;height:2.1rem;border-radius:50%;background:#176B5B;color:#fff;
      display:flex;align-items:center;justify-content:center;font-weight:700;font-size:.95rem}
.job-title{font-size:1.1rem;font-weight:650;color:#172B3A;line-height:1.3}
.job-meta{color:#5b6f80;font-size:.88rem;margin-top:.15rem}
.chips{margin:.6rem 0 .2rem}
.chip{display:inline-block;padding:2px 11px;margin:0 6px 6px 0;border-radius:999px;font-size:.8rem;
      background:#E3F1EE;color:#0f4c41;border:1px solid #bfe0d8}
.chip.muted{background:#EEF2F6;color:#41576a;border-color:#d8e0e8}
.chip.money{background:#FFF4DB;color:#7a5200;border-color:#f1dba4}
.desc{color:#2c4254;font-size:.95rem;line-height:1.5;margin:.35rem 0 .2rem}
.step{background:#fff;border:1px solid #dbe4ec;border-radius:14px;padding:1rem 1.1rem;height:100%}
.step b{color:#176B5B}
.step-n{display:inline-flex;width:1.8rem;height:1.8rem;border-radius:50%;background:#E3F1EE;color:#176B5B;
        align-items:center;justify-content:center;font-weight:700;margin-bottom:.4rem}
.foot{color:#5b6f80;font-size:.82rem;margin-top:1rem}
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

st.markdown(
    '<div class="hero"><div class="hero-title">Find jobs that fit your experience</div>'
    '<div class="hero-sub">Describe your role, background and skills. The system ranks vacancies from a '
    'historical Russian labour-market dataset and shows them in English.</div>'
    '<div class="hero-tags"><span>Hybrid text + meaning matching</span><span>Skill overlap shown per job</span>'
    '<span>Runs locally, nothing is saved</span></div></div>', unsafe_allow_html=True)


@st.cache_resource(show_spinner='Loading the recommendation model…')
def load_model(path, manifest_signature):
    from recommender.serving import Recommender
    return Recommender(path)


@st.cache_resource(show_spinner='Loading English language support…')
def load_english(path, manifest_signature):
    from recommender.translation import EnglishAdapter
    adapter = EnglishAdapter(path)
    adapter.warm_up()
    return adapter


try:
    bundle = check_bundle()
    model = load_model(str(bundle), (bundle / 'manifest.json').read_bytes())
    translation = check_translation()
    english = load_english(str(translation), (translation / 'manifest.json').read_bytes())
    region_labels = english.region_labels(model.regions)
except (OSError, ValueError) as error:
    st.error(str(error))
    st.stop()

available_skills = skill_options(model.extractor.dictionary)
skill_order = sorted(available_skills, key=lambda sid: SKILL_LABELS[sid])
region_order = [''] + sorted(model.regions, key=lambda r: region_labels[r])
ROLE_PLACEHOLDER = 'Choose a job title'
OTHER = 'Other (type your own)'

DEFAULTS = {'role': ROLE_PLACEHOLDER, 'custom_role': '', 'experience': EXPERIENCE[0], 'experience_other': '',
            'education': EDUCATION[0], 'education_other': '', 'responsibilities': [], 'resp_other': '',
            'skills': [], 'skills_other': '', 'notes': '', 'region': '', 'region_other': '',
            'limit_text': '10', 'group_titles': False}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def reset_form():
    for key, value in DEFAULTS.items():
        st.session_state[key] = value
    st.session_state.pop('result', None)
    st.session_state.pop('error', None)
    st.session_state['f_salary'] = False
    st.session_state['f_min'] = 0


def esc(value):
    return html.escape(str(value), quote=True)


def chips(items, css=''):
    return ''.join(f'<span class="chip {css}">{esc(item)}</span>' for item in items)


def salary_text(value):
    try:
        number = float(str(value).replace(',', '').strip())
    except ValueError:
        return None
    return f'{number:,.0f} RUB' if number > 0 else None


def split_list(text):
    return [part.strip() for part in re.split(r'[;,\n]', text) if part.strip()]


def resolve_region(text):
    """Match a typed region to a catalog region; return (key, note). Unmatched text is kept as typed."""
    value = ' '.join(text.lower().replace('-', ' ').split())
    if not value:
        return '', None
    def norm(name):
        return ' '.join(str(name).lower().replace('-', ' ').split())
    for key in model.regions:
        if value in (norm(key), norm(region_labels[key])):
            return key, None
    close = [key for key in model.regions if norm(region_labels[key]).startswith(value)]
    if len(close) == 1:
        return close[0], None
    return text.strip(), ('“' + text.strip() + '” was not found in the catalog, so no local region priority '
                          'was applied. Pick a region from the list for local priority.')


def parse_limit(text):
    try:
        number = int(str(text).strip())
    except ValueError:
        raise ValueError('Number of suggestions must be a whole number between 1 and 20.')
    if not 1 <= number <= 20:
        raise ValueError('Number of suggestions must be between 1 and 20.')
    return number


def shorten(text, limit=420):
    text = ' '.join(str(text).split())
    if len(text) <= limit:
        return text, False
    return text[:limit].rsplit(' ', 1)[0] + '…', True


with st.sidebar:
    st.markdown('### About')
    st.metric('Vacancies searched', f'{len(model.catalog):,}')
    st.caption('Model: ' + model.meta['selected_model'] + ' · historical project dataset')
    st.write('Results are suggestions, not hiring predictions. Check posting availability and requirements before applying.')
    st.caption('Your profile is translated and processed locally in memory; it is not saved by this app.')
    st.caption('Vacancy text is machine-translated into English. Names and specialist terms may contain '
               'translation errors. Salaries are listed in Russian rubles (RUB).')
    st.caption(VERSION_LINE)

left, right = st.columns([5, 7], gap='large')

with left:
    with st.container(border=True):
        st.markdown('<div class="sec">1 · Role</div>', unsafe_allow_html=True)
        role = st.selectbox('Desired job title', [ROLE_PLACEHOLDER] + JOB_TITLES + ['Other'], key='role')
        if role == 'Other':
            st.text_input('Your job title', max_chars=300, key='custom_role',
                          placeholder='For example: Quality inspector')
        st.markdown('<div class="sec">2 · Background</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.selectbox('Work experience', EXPERIENCE, key='experience')
        c2.selectbox('Education', EDUCATION, key='education')
        st.multiselect('Previous responsibilities', RESPONSIBILITIES, key='responsibilities',
                       help='Choose only responsibilities you have performed.',
                       placeholder='Select all that apply')
        st.text_input('Other responsibilities (optional)', max_chars=500, key='resp_other',
                      placeholder='Not in the list? Type them, separated by commas')
        st.markdown('<div class="sec">3 · Skills</div>', unsafe_allow_html=True)
        st.multiselect('Your skills', skill_order, key='skills', format_func=lambda sid: SKILL_LABELS[sid],
                       help='These fixed choices map directly to the project skill dictionary.',
                       placeholder='Search and select skills')
        st.text_input('Other skills (optional)', max_chars=500, key='skills_other',
                      placeholder='Not in the list? Type them, separated by commas',
                      help='Typed skills help the ranking through the text, but are not counted in the matched-skill tags.')
        st.text_area('Anything else? (optional)', max_chars=3000, height=80, key='notes',
                     placeholder='Add relevant details not covered by the choices above.')
        st.markdown('<div class="sec">4 · Preferences</div>', unsafe_allow_html=True)
        st.selectbox('Preferred region', region_order + [OTHER], key='region',
                     format_func=lambda r: OTHER if r == OTHER else (region_labels[r] if r else 'Any region / not specified'))
        if st.session_state['region'] == OTHER:
            st.text_input('Your region', max_chars=200, key='region_other', placeholder='For example: Moscow')
        st.text_input('Number of suggestions (1–20)', max_chars=2, key='limit_text')
        st.checkbox('One posting per job title and region', key='group_titles',
                    help='This can hide different employers with the same title. Leave unchecked to see separate vacancies.')
        b1, b2 = st.columns([3, 1])
        search = b1.button('Find jobs', type='primary', use_container_width=True)
        b2.button('Clear', on_click=reset_form, use_container_width=True)

if search:
    st.session_state.pop('error', None)
    try:
        s = st.session_state
        limit = parse_limit(s['limit_text'])
        experience = s['experience']
        education = s['education']
        responsibilities = list(s['responsibilities']) + split_list(s['resp_other'])
        extra_skills = split_list(s['skills_other'])
        notes = s['notes'].strip()
        region, region_note = (resolve_region(s['region_other']) if s['region'] == OTHER else (s['region'], None))
        with right:
            with st.spinner('Translating your profile and finding matching vacancies…'):
                title, history, qualification = build_profile(
                    s['role'], s['custom_role'], experience, education, responsibilities, notes)
                ru_title, ru_history, ru_education, ru_typed = english.profile(
                    title, history, qualification, ', '.join(extra_skills))
                # Skills enter the model once, as cv_skills, exactly as in the evaluated data.
                # Listed skills keep their Russian dictionary names; typed skills are translated.
                ru_skills = '; '.join([available_skills[sid]['canonical'] for sid in s['skills']]
                                      + ([ru_typed] if ru_typed.strip() else []))
                result, info = model.recommend(ru_title, ru_history, ru_education, ru_skills,
                                               region, limit, s['group_titles'])
            with st.spinner('Preparing your results in English…'):
                display = english.results(result)
                matches = exact_skill_matches(model, result, s['skills'])
                display['Matched skill count'] = [len(names) for names in matches]
                display['Matched skills'] = ['; '.join(names) for names in matches]
        s['result'] = {'display': display, 'matches': matches, 'info': info, 'title': title,
                       'chosen': len(s['skills']), 'region_note': region_note}
        s['f_salary'] = False
        s['f_min'] = 0
    except (OSError, ValueError, RuntimeError) as error:
        st.session_state['error'] = str(error)
        st.session_state.pop('result', None)

with right:
    saved = st.session_state.get('result')
    if st.session_state.get('error'):
        st.warning(st.session_state['error'])
    if not saved:
        if not st.session_state.get('error'):
            st.markdown('#### How it works')
            s1, s2, s3 = st.columns(3)
            s1.markdown('<div class="step"><div class="step-n">1</div><br><b>Tell us about you</b><br>'
                        'Pick a job title, your experience and the skills you have.</div>', unsafe_allow_html=True)
            s2.markdown('<div class="step"><div class="step-n">2</div><br><b>We rank vacancies</b><br>'
                        'Text similarity and meaning similarity are combined, with your region first.</div>',
                        unsafe_allow_html=True)
            s3.markdown('<div class="step"><div class="step-n">3</div><br><b>Review the matches</b><br>'
                        'Each job lists the skills you share with it. Download the list as CSV.</div>',
                        unsafe_allow_html=True)
            st.markdown('')
            st.info('Fill in the form on the left and press **Find jobs**.')
    else:
        display, matches, info = saved['display'], saved['matches'], saved['info']
        rows = list(zip(display.to_dict('records'), matches))
        listed = sum(1 for row, _ in rows if salary_text(row['Listed minimum salary (RUB)']))
        best = max((len(names) for _, names in rows), default=0)

        if saved.get('region_note'):
            st.info(saved['region_note'])
        elif info['region_priority'] and not info['known_region']:
            st.info('No region was selected. Vacancies across the catalog are shown; choose a region for local priority.')

        h1, h2 = st.columns([3, 2])
        h1.markdown(f"#### Suggested vacancies for “{esc(saved['title'])}”", unsafe_allow_html=True)
        h2.download_button('Download all as CSV', display.to_csv(index=False).encode('utf-8-sig'),
                           file_name='job_suggestions_english.csv', mime='text/csv', use_container_width=True)
        m1, m2, m3 = st.columns(3)
        m1.metric('Suggestions', len(rows))
        m2.metric('With salary listed', f'{listed} of {len(rows)}')
        m3.metric('Best skill overlap', f"{best} of {saved['chosen']}" if saved['chosen'] else 'n/a')

        f1, f2 = st.columns(2)
        only_salary = f1.checkbox('Only vacancies with a listed salary', key='f_salary')
        min_match = f2.slider('Minimum matched skills', 0, max(best, 1), key='f_min', disabled=best == 0)
        shown = [(row, names) for row, names in rows
                 if len(names) >= min_match and (not only_salary or salary_text(row['Listed minimum salary (RUB)']))]
        st.caption('Matched skills count your selected dictionary skills with positive vacancy evidence. '
                   'A zero count can reflect limited dictionary coverage; it is not proof of incompatibility.')
        if not shown:
            st.info('No vacancies match these filters. Relax them to see all suggestions.')

        for row, names in shown:
            with st.container(border=True):
                meta = ['📍 ' + str(row['Region'])]
                published = str(row.get('Published', '') or '').strip()
                if len(published) >= 10:
                    meta.append('Published ' + published[:10])
                st.markdown(
                    f'<div class="job-head"><div class="rank">{esc(row["Rank"])}</div><div>'
                    f'<div class="job-title">{esc(row["Job title"])}</div>'
                    f'<div class="job-meta">{esc(" · ".join(meta))}</div></div></div>', unsafe_allow_html=True)
                tags = []
                pay = salary_text(row['Listed minimum salary (RUB)'])
                tags.append(chips([('From ' + pay) if pay else 'Salary not specified'], 'money' if pay else 'muted'))
                tags.append(chips([str(row['Required experience'])], 'muted'))
                if row['Professional field']:
                    tags.append(chips([str(row['Professional field'])], 'muted'))
                st.markdown('<div class="chips">' + ''.join(tags) + '</div>', unsafe_allow_html=True)
                if names:
                    st.markdown('<div class="chips"><b style="font-size:.82rem;color:#176B5B">'
                                + esc(match_caption(names)) + ':</b> ' + chips(names) + '</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown('<div class="chips">' + chips([match_caption(names)], 'muted') + '</div>',
                                unsafe_allow_html=True)
                short, truncated = shorten(row['Description'])
                st.markdown(f'<div class="desc">{esc(short)}</div>', unsafe_allow_html=True)
                if truncated or row['Skills']:
                    with st.expander('Description and requirements'):
                        st.write(row['Description'])
                        if row['Skills']:
                            st.write('**Skills:**', row['Skills'])
                st.caption('Vacancy reference: ' + str(row['Vacancy reference']))
        st.markdown('<div class="foot">Salaries are the listed minimum in Russian rubles. Descriptions are '
                    'machine-translated and may contain errors.</div>', unsafe_allow_html=True)

st.markdown(f'<div class="foot">{esc(VERSION_LINE)}</div>', unsafe_allow_html=True)
