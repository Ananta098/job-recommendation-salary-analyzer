"""English Streamlit interface backed by the evaluated Russian M8 model."""
import streamlit as st
from recommender.bundle import check_bundle
from recommender.translation import check_translation
from recommender.profile_form import (JOB_TITLES, EXPERIENCE, EDUCATION, RESPONSIBILITIES,
    SKILL_LABELS, skill_options, build_profile, exact_skill_matches, match_caption)

st.set_page_config(page_title='Job Recommendation System',page_icon='💼',layout='wide')
st.title('Find jobs that fit your experience')
st.write('Choose your job preferences and skills to explore vacancies from the historical Russian labour-market dataset.')

@st.cache_resource(show_spinner='Loading the recommendation model…')
def load_model(path, manifest_signature):
    from recommender.serving import Recommender
    return Recommender(path)

@st.cache_resource(show_spinner='Loading English language support…')
def load_english(path, manifest_signature):
    from recommender.translation import EnglishAdapter
    return EnglishAdapter(path)

try:
    bundle=check_bundle()
    model=load_model(str(bundle), (bundle/'manifest.json').read_bytes())
    translation=check_translation()
    english=load_english(str(translation),(translation/'manifest.json').read_bytes())
    region_labels=english.region_labels(model.regions)
except (OSError,ValueError) as error:
    st.error(str(error));st.stop()

with st.sidebar:
    st.caption('Model: '+model.meta['selected_model'])
    st.caption(f'{len(model.catalog):,} vacancies · historical project dataset')
    st.write('Results are suggestions, not hiring predictions. Check posting availability and requirements before applying.')
    st.caption('Your profile is translated and processed locally in memory; it is not saved by this app.')
    st.caption('Vacancy text is machine-translated into English. Names and specialist terms may contain translation errors. Salaries are listed in Russian rubles (RUB).')

with st.form('profile'):
    role=st.selectbox('Desired job title',['Choose a job title']+JOB_TITLES+['Other'])
    custom_role=st.text_input('Custom job title (only for Other)',max_chars=300,placeholder='Enter a title if it is not in the list.')
    experience=st.selectbox('Work experience',EXPERIENCE)
    education=st.selectbox('Highest education or qualification',EDUCATION)
    responsibilities=st.multiselect('Previous responsibilities',RESPONSIBILITIES,help='Choose only responsibilities you have performed.')
    available_skills=skill_options(model.extractor.dictionary)
    selected_skills=st.multiselect('Your skills',sorted(available_skills,key=lambda sid:SKILL_LABELS[sid]),format_func=lambda sid:SKILL_LABELS[sid],help='Choose skills you have. These fixed choices map directly to the project skill dictionary.')
    notes=st.text_area('Additional experience or qualifications (optional)',max_chars=3000,height=90,placeholder='Add relevant details not covered by the choices above.')
    region=st.selectbox('Preferred region',['']+sorted(model.regions,key=lambda r:region_labels[r]),format_func=lambda r:region_labels[r] if r else 'Any region / not specified')
    limit=st.slider('Number of suggestions',5,20,10)
    group_titles=st.checkbox('Show one posting per job title and region',value=False,help='This can hide different employers with the same title. Leave unchecked to see separate vacancies.')
    submitted=st.form_submit_button('Find jobs',type='primary')

if submitted:
    try:
        with st.spinner('Translating your profile and finding matching vacancies…'):
            title,history,qualification=build_profile(role,custom_role,experience,education,responsibilities,selected_skills,notes)
            ru_title,ru_history,ru_education,_=english.profile(title,history,qualification,'')
            # Preserve dictionary aliases instead of translating skill names approximately.
            ru_skills='; '.join(available_skills[sid]['canonical'] for sid in selected_skills)
            if ru_skills:ru_history+=' '+ru_skills
            result,info=model.recommend(ru_title,ru_history,ru_education,ru_skills,region,limit,group_titles)
        with st.spinner('Preparing your results in English…'):
            display=english.results(result)
            matches=exact_skill_matches(model,result,selected_skills)
            display['Matched skill count']=[len(names) for names in matches]
            display['Matched skills']=['; '.join(names) for names in matches]
        if info['region_priority'] and not info['known_region']:
            st.info('No region was selected. Vacancies across the catalog are shown; choose a region for local priority.')
        st.subheader('Suggested vacancies')
        st.caption('Matched skills count your selected dictionary skills with positive vacancy evidence. A zero count can reflect limited dictionary coverage; it is not proof of incompatibility.')
        for row,names in zip(display.to_dict('records'),matches):
            with st.expander(f"{row['Rank']}. {row['Job title']} — {row['Region']}",expanded=row['Rank']==1):
                st.markdown('**'+match_caption(names)+'**')
                if names:st.write('Matched skills:',', '.join(names))
                salary=row['Listed minimum salary (RUB)']
                if str(salary).strip() and str(salary) not in ('0','0.0','nan'):
                    st.write('Listed minimum salary (RUB):',str(salary))
                else:st.write('Listed minimum salary: Not specified')
                st.write('Required experience:',row['Required experience'])
                st.write(row['Description'])
                if row['Skills']:st.write('Skills:',row['Skills'])
                if row['Professional field']:st.write('Professional field:',row['Professional field'])
                st.caption('Vacancy reference: '+str(row['Vacancy reference']))
        st.download_button('Download suggestions as CSV',display.to_csv(index=False).encode('utf-8-sig'),file_name='job_suggestions_english.csv',mime='text/csv')
    except (OSError,ValueError,RuntimeError) as error:
        st.warning(str(error))
