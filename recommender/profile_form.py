"""Structured English profile choices and exact canonical skill explanations."""
JOB_TITLES = ['Accountant','Administrative assistant','Business analyst','Cashier',
              'Cleaner','Cook','Customer service representative','Data analyst',
              'Database administrator','Delivery driver','Doctor','Electrician',
              'Engineer','HR specialist','IT support specialist','Laboratory technician',
              'Lawyer','Machine operator','Maintenance technician','Mechanic','Nurse',
              'Office administrator','Procurement specialist','Project manager',
              'Sales representative','Security guard','Software developer',
              'Teacher','Warehouse worker','Welder']
EXPERIENCE = ['Not specified','No previous work experience','Less than 1 year',
              '1–3 years','3–5 years','5–10 years','More than 10 years']
EDUCATION = ['Not specified','Secondary school','Vocational or technical qualification',
             'College diploma','Bachelor’s degree','Master’s degree','Doctoral degree']
RESPONSIBILITIES = ['Bookkeeping and financial reporting','Payroll and tax reporting',
    'Customer service','Sales and client communication','Office administration and records',
    'Procurement and inventory management','Software development','Data analysis and reporting',
    'Equipment repair and maintenance','Electrical installation and maintenance',
    'Driving and deliveries','Teaching and lesson planning','Patient care and medical records',
    'Cooking and food preparation','Cleaning and sanitation','Welding and fabrication',
    'Project coordination','Staff recruitment and HR records','Cash handling']
SKILL_LABELS = {
 'computer':'Computer literacy','office':'Office software','excel':'Microsoft Excel',
 'word':'Microsoft Word','1c':'1C accounting software','accounting':'Bookkeeping',
 'tax_law':'Tax legislation','tax_reporting':'Tax reporting',
 'financial_reporting':'Financial reporting','hr_records':'HR records',
 'document_handling':'Document handling','teaching_plan':'Lesson planning',
 'teaching':'Teaching','parents':'Parent communication',
 'teaching_assessment':'Student assessment','sanitary':'Sanitation standards',
 'driving':'Driving','traffic_rules':'Traffic rules',
 'client_needs':'Identifying customer needs',
 'electrical_maintenance':'Electrical equipment maintenance',
 'teamwork':'Teamwork','communication':'Communication','stress':'Stress tolerance',
 'attention':'Attention to detail','python':'Python','sql':'SQL','cpp':'C++',
 'csharp':'C#','dotnet':'.NET','welding':'Welding','cleaning':'Cleaning',
 'procurement':'Procurement','labor_law':'Labour law',
 'business_correspondence':'Business correspondence',
 'records_management':'Records management','online_banking':'Online banking',
 'tax_accounting':'Tax accounting','statistical_reporting':'Statistical reporting',
 'office_equipment':'Office equipment','active_sales':'Active sales',
 'cash_register':'Cash register operation','first_aid':'First aid','cooking':'Cooking',
 'drawing_reading':'Technical drawing reading','fire_safety':'Fire safety',
 'medical_records':'Medical records','occupational_safety':'Occupational safety',
}


def skill_options(dictionary):
    # Match the ordering/eligibility contract of the exported job skill matrix.
    return {s['id']:s for s in dictionary['skills'] if s['domain']!='soft'}


def build_profile(role, custom_role, experience, education, responsibilities, selected, notes):
    title = custom_role.strip() if role=='Other' else role
    if not title or role=='Choose a job title':
        raise ValueError('Choose a job title, or choose Other and enter a custom title.')
    history = []
    if experience!='Not specified':history.append('Work experience: '+experience+'.')
    if responsibilities:history.append('Responsibilities: '+ '; '.join(responsibilities)+'.')
    # M8 consumes work-history text; structured skills must reach that text too.
    if selected:history.append('Skills: '+', '.join(SKILL_LABELS[s] for s in selected)+'.')
    if notes.strip():history.append(notes.strip())
    qualification = '' if education=='Not specified' else education
    return title, ' '.join(history), qualification


def exact_skill_matches(model, result, selected):
    """Count selected canonical IDs with positive stored vacancy evidence once each.

    Negated/non-required mentions and soft traits were excluded during export.
    This explains dictionary overlap, not a probability or a new ranking score.
    """
    skills = list(skill_options(model.extractor.dictionary))
    columns = {sid:i for i,sid in enumerate(skills)}
    index = {str(vid):i for i,vid in enumerate(model.catalog.idVacancy)}
    chosen = sorted(set(selected).intersection(columns),key=lambda sid:SKILL_LABELS[sid])
    return [[SKILL_LABELS[sid] for sid in chosen
             if model.skills[index[str(vid)],columns[sid]]>0]
            for vid in result.idVacancy]


def match_caption(names):
    count=len(names)
    return f'{count} matched skill' + ('' if count==1 else 's')
