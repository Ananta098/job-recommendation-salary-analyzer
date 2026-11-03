"""Static diagnostic plots. Requires matplotlib, already in project requirements."""
import argparse
import json
from pathlib import Path

def plot(results, baseline_coverage, expanded_coverage, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    reports={name:json.loads((Path(results)/(name+'.json')).read_text()) for name in ('baseline','expanded')}
    before=json.loads(Path(baseline_coverage).read_text());after=json.loads(Path(expanded_coverage).read_text())
    colors=['#7893a9','#008a7e'];names=[f"Original ({reports['baseline']['canonical_skills']} skills)",f"Expanded ({reports['expanded']['canonical_skills']} skills)"]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(14,10));fig.subplots_adjust(top=.87,bottom=.14,hspace=.52,wspace=.30)
    fig.suptitle('Russian skill extraction: validation diagnostics',fontsize=20,fontweight='bold',y=.97)
    fig.text(.5,.925,f"Diagnostic annotations • {reports['expanded']['human_reviewed_cases']} human-reviewed cases • v2 remains opt-in",ha='center',color='#525d67',fontsize=12)
    for ax,suite,title in zip(axes[0],['corpus_diagnostic','authored_challenge'],['Real corpus phrases (39 scored; 1 ambiguous excluded)','Authored challenge cases (40; dictionary-informed)']):
        x=np.arange(3)
        for j,name in enumerate(('baseline','expanded')):
            m=reports[name]['by_suite'][suite];values=[100*m[k] for k in ('precision','recall','f1')]
            bars=ax.bar(x+(j-.5)*.34,values,width=.33,color=colors[j],label=names[j])
            ax.bar_label(bars,fmt='%.1f%%',fontsize=9,padding=3)
        ax.set_xticks(x,['Precision','Recall','F1']);ax.set_ylim(0,115);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel('Closed-inventory score (%)');ax.set_title(title,fontsize=12,pad=14);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axes[0,0].legend(fontsize=9,loc='lower left')
    ax=axes[1,0];x=np.arange(2)
    for j,cov in enumerate((before,after)):
        vals=[100*cov['diagnostics'][s]['mention_coverage'] for s in ('cv','vacancy')]
        bars=ax.bar(x+(j-.5)*.34,vals,width=.33,color=colors[j]);ax.bar_label(bars,fmt='%.2f%%',fontsize=10,padding=3)
    ax.set_xticks(x,['65,714 resumes','43,756 vacancies']);ax.set_ylim(0,100);ax.set_ylabel('Entities with ≥1 recognized skill (%)');ax.set_title('Full-data mention coverage — not accuracy',fontsize=12,pad=14);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    ax=axes[1,1];x=np.arange(3)
    for j,name in enumerate(('baseline','expanded')):
        suites=reports[name]['by_suite'];vals=[sum(m[k] for m in suites.values()) for k in ('false_positive','false_negative','assertion_error_cases')]
        bars=ax.bar(x+(j-.5)*.34,vals,width=.33,color=colors[j]);ax.bar_label(bars,fmt='%d',fontsize=10,padding=3)
    ax.set_xticks(x,['Extra skill labels','Missed skill labels','Assertion-error cases']);ax.set_ylim(0,32);ax.set_ylabel('Count across both diagnostic suites');ax.set_title('Unresolved errors remain visible',fontsize=12,pad=14);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.text(.06,.075,'Read with care: 26 corpus cases contain out-of-inventory activities or traits; not all should become skills.',fontsize=11,color='#8b4e15')
    fig.text(.06,.048,'Scores use agent gold labels within that inventory. Small, enriched diagnostics do not establish population accuracy.',fontsize=10,color='#525d67')
    fig.text(.06,.023,'Separate human review and open-vocabulary assessment are still required. No M9 ranking evaluation is included.',fontsize=10,color='#525d67')
    fig.savefig(output/'validation_overview.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    return str(output/'validation_overview.png')

def plot_spheres(rows, output):
    import textwrap
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=rows[:8]
    fig,ax=plt.subplots(figsize=(13,8));fig.subplots_adjust(left=.42,right=.95,top=.84,bottom=.16)
    for i,r in enumerate(rows):
        a=r['baseline_coverage']*100;b=r['expanded_coverage']*100
        ax.plot([a,b],[i,i],color='#a3b4be',linewidth=3,zorder=1)
        ax.scatter(a,i,color='#7893a9',s=65,label='Original' if i==0 else None,zorder=2)
        ax.scatter(b,i,color='#008a7e',s=65,label='Expanded' if i==0 else None,zorder=3)
        ax.text(max(a,b)+2,i,f'{b:.1f}%',va='center',fontsize=10)
    ax.set_yticks(range(len(rows)),['\n'.join(textwrap.wrap(r['sphere'],38))+'  (n='+str(r['vacancies'])+')' for r in rows],fontsize=10)
    ax.set_xlim(0,100);ax.set_ylim(len(rows)-.4,-.6);ax.set_xlabel('Unique vacancies with ≥1 recognized skill (%)')
    ax.spines[['top','right','left']].set_visible(False);ax.grid(axis='x',alpha=.15);ax.tick_params(axis='y',length=0)
    ax.legend(loc='upper right',fontsize=10)
    fig.suptitle('Coverage varies across occupational areas',fontsize=18,fontweight='bold',y=.96)
    fig.text(.5,.90,'Eight largest professional spheres • all 43,756 unique vacancies included in source table',ha='center',fontsize=11,color='#525d67')
    fig.text(.06,.065,'Coverage counts recognized mentions, not suitability or extraction accuracy. Denominators are unique vacancy IDs.',fontsize=10,color='#525d67')
    fig.text(.06,.035,'The expanded dictionary improves some areas; substantial vocabulary gaps remain. Human review is pending.',fontsize=10,color='#525d67')
    fig.savefig(output/'coverage_by_sphere.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    return str(output/'coverage_by_sphere.png')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',required=True);p.add_argument('--baseline-coverage',required=True);p.add_argument('--expanded-coverage',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(plot(a.results,a.baseline_coverage,a.expanded_coverage,a.output))
