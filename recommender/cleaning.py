"""Original D4 cleaning, factored from notebook 02; preserves the evaluated policy.

Entity snapshots still follow invitation-first deduplication. This is documented
as a limitation, not a chronological or label-independent preprocessing claim.
"""
from pathlib import Path as _Path
from contextlib import chdir

def clean_data(root):
    root = _Path(root).resolve()
    with chdir(root):
        import os
        import re
        import json
        from pathlib import Path

        import numpy as np
        import pandas as pd

        # Work from the project folder, even when Jupyter opens inside notebooks/

        RAW_FILE    = "datasets/HSE_dataset(D4).csv"
        LOOKUP_FILE = "lookup/job_titles_ru_en.csv"
        OUT_FILE    = "clean/d4_clean.csv"

        # If the raw file has another name, take the csv in datasets/ that has "hse" in its name
        if not Path(RAW_FILE).exists():
            found = [p for p in Path("datasets").glob("*.csv") if "hse" in p.name.lower()]
            if not found:
                raise FileNotFoundError("Put the HSE csv file in the datasets/ folder")
            RAW_FILE = str(found[0])

        os.makedirs("clean", exist_ok=True)
        os.makedirs("results", exist_ok=True)

        steps = []   # a short log of each step, saved at the end for the report

        print("project folder:", os.getcwd())
        print("raw file      :", RAW_FILE)

        SEP = "|"

        USE_COLS = [
            "idCv", "idVacancy", "cv_status",
            # applicant side
            "positionName", "workExperienceList", "educationList",
            "skills_cv", "hardSkills_cv", "softSkills_cv",
            "experience", "salaryMin_cv", "localityName",
            # job side
            "vacancyName", "responsibilities", "positionRequirements",
            "skills_vacancy", "hardSkills_vacancy", "softSkills_vacancy",
            "professionalSphereName", "experienceRequirements", "salaryMin_vacancy",
            "regionName", "datePublished",
        ]

        header = pd.read_csv(RAW_FILE, sep=SEP, nrows=0).columns
        missing = [c for c in USE_COLS if c not in header]
        USE_COLS = [c for c in USE_COLS if c in header]

        print("columns in raw file :", len(header))
        print("columns we load     :", len(USE_COLS))
        print("not found           :", missing if missing else "none")

        df = pd.read_csv(RAW_FILE, sep=SEP, usecols=USE_COLS, low_memory=False)

        print("rows loaded         :", f"{len(df):,}")
        steps.append(("1. Load", f"{len(df):,} rows, {len(USE_COLS)} of {len(header)} columns"))

        df["is_positive"] = (df["cv_status"] == "Приглашение").astype(int)

        before = len(df)

        # invitations first, then keep the first row of each (applicant, job) pair
        df = df.sort_values("is_positive", ascending=False)
        df = df.drop_duplicates(subset=["idCv", "idVacancy"], keep="first")
        df = df.reset_index(drop=True)

        removed = before - len(df)
        print(f"rows: {before:,} -> {len(df):,}")
        print(f"removed {removed:,} duplicate rows ({removed / before:.1%})")
        print(f"invitation rate now: {df['is_positive'].mean():.1%}")

        steps.append(("2. Remove duplicates", f"{before:,} -> {len(df):,} rows ({removed / before:.1%} removed)"))

        def fix_hyphens(text):
            # 'Специалист-по-кадрам' -> 'Специалист по кадрам'
            if not isinstance(text, str):
                return text
            return " ".join(text.replace("-", " ").split())

        def shares_a_word(a, b):
            words_a = set(str(a).lower().split())
            words_b = set(str(b).lower().split())
            return len(words_a & words_b) > 0

        sample = df.head(5000)
        share_before = sample.apply(lambda r: shares_a_word(r["positionName"], r["vacancyName"]), axis=1).mean()

        df["positionName"] = df["positionName"].map(fix_hyphens)
        df["localityName"] = df["localityName"].map(fix_hyphens)

        sample = df.head(5000)
        share_after = sample.apply(lambda r: shares_a_word(r["positionName"], r["vacancyName"]), axis=1).mean()

        print(f"applicant title and job title share a word: {share_before:.1%} -> {share_after:.1%}")

        steps.append(("3. Fix hyphens", f"title word overlap {share_before:.1%} -> {share_after:.1%}"))

        def read_records(cell, keys):
            # [{"jobTitle": "A", "demands": "B"}, ...] -> "A B ..."
            try:
                items = json.loads(cell)
            except (TypeError, ValueError):
                return ""
            if not isinstance(items, list):
                return ""
            words = []
            for item in items:
                if isinstance(item, dict):
                    for key in keys:
                        value = item.get(key)
                        if value and value != "None":
                            words.append(str(value))
            return " ".join(words)

        def read_skills(cell):
            # ["Работа в 1С", "Excel"] -> "Работа в 1С Excel"
            try:
                items = json.loads(cell)
            except (TypeError, ValueError):
                return ""
            if not isinstance(items, list):
                return ""
            return " ".join(str(x) for x in items if x)

        df["exp_text"] = df["workExperienceList"].map(lambda c: read_records(c, ["jobTitle", "demands"]))
        df["edu_text"] = df["educationList"].map(lambda c: read_records(c, ["speciality", "qualification", "instituteName"]))

        # join the three skill columns on each side into one text
        df["cv_skills"] = ""
        for col in ["skills_cv", "hardSkills_cv", "softSkills_cv"]:
            if col in df.columns:
                df["cv_skills"] = df["cv_skills"] + " " + df[col].map(read_skills)

        df["job_skills"] = ""
        for col in ["skills_vacancy", "hardSkills_vacancy", "softSkills_vacancy"]:
            if col in df.columns:
                df["job_skills"] = df["job_skills"] + " " + df[col].map(read_skills)

        df["cv_skills"] = df["cv_skills"].str.strip()
        df["job_skills"] = df["job_skills"].str.strip()

        print(f"rows with work history text : {(df['exp_text'] != '').mean():.1%}")
        print(f"rows with education text    : {(df['edu_text'] != '').mean():.1%}")
        print()

        applicant_has_skill = df.groupby("idCv")["cv_skills"].first() != ""
        job_has_skill       = df.groupby("idVacancy")["job_skills"].first() != ""

        print(f"applicants with at least one skill: {applicant_has_skill.mean():.1%}  ({applicant_has_skill.sum():,} people)")
        print(f"jobs with at least one skill      : {job_has_skill.mean():.1%}  ({job_has_skill.sum():,} jobs)")

        steps.append(("4. Skills", f"{applicant_has_skill.mean():.1%} of applicants, {job_has_skill.mean():.1%} of jobs have skills"))

        def join_parts(*parts):
            good = [str(p) for p in parts if isinstance(p, str) and p.strip() and p != "nan"]
            return " ".join(good)

        def normalise(text):
            text = str(text).lower()
            text = re.sub(r"[^a-zа-яё0-9\s]", " ", text)   # keep letters and digits only
            return " ".join(text.split())

        df["cv_text"]  = [join_parts(a, b, c) for a, b, c in zip(df["positionName"], df["exp_text"], df["edu_text"])]
        df["job_text"] = [join_parts(a, b, c) for a, b, c in zip(df["vacancyName"], df["responsibilities"], df["positionRequirements"])]

        for col in ["cv_text", "job_text", "cv_skills", "job_skills"]:
            df[col] = df[col].map(normalise)

        print()
        print(f"average words  cv_text: {df['cv_text'].str.split().str.len().mean():.0f}   "
              f"job_text: {df['job_text'].str.split().str.len().mean():.0f}")

        steps.append(("5. Build texts", "cv_text and job_text, lowercased, punctuation removed"))

        APPLICANT_COLS = ["positionName", "cv_text", "cv_skills", "experience", "salaryMin_cv", "localityName"]
        JOB_COLS = ["vacancyName", "job_text", "job_skills", "professionalSphereName",
                    "experienceRequirements", "salaryMin_vacancy", "regionName", "datePublished"]

        for col in APPLICANT_COLS:
            df[col] = df.groupby("idCv")[col].transform("first")

        for col in JOB_COLS:
            df[col] = df.groupby("idVacancy")[col].transform("first")

        print("done")

        for col in ["experience", "experienceRequirements", "salaryMin_cv", "salaryMin_vacancy"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            n_missing = df[col].isna().sum()
            df[col] = df[col].fillna(df[col].median())
            print(f"{col:24s} filled {n_missing:,} missing values")

        for col in ["professionalSphereName", "regionName", "localityName"]:
            df[col] = df[col].fillna("Unknown")

        df["datePublished"] = pd.to_datetime(df["datePublished"], errors="coerce", format="mixed", utc=True)

        steps.append(("6. Missing values", "numbers filled with the median, text filled with Unknown"))

        def title_key(text):
            # match titles even if case, hyphens or spaces differ
            return " ".join(str(text).lower().replace("-", " ").split())

        lookup = pd.read_csv(LOOKUP_FILE, encoding="utf-8-sig")
        ru_to_en = {title_key(ru): en for ru, en in zip(lookup["ru"], lookup["en"])}

        df["positionName_en"] = [ru_to_en.get(title_key(t), t) for t in df["positionName"]]
        df["vacancyName_en"]  = [ru_to_en.get(title_key(t), t) for t in df["vacancyName"]]

        cover_cv  = (df["positionName_en"] != df["positionName"]).mean()
        cover_job = (df["vacancyName_en"] != df["vacancyName"]).mean()
        print(f"titles in the list: {len(lookup):,}")
        print(f"rows with English applicant title: {cover_cv:.1%}")
        print(f"rows with English job title      : {cover_job:.1%}")

        steps.append(("7. English titles", f"{cover_job:.1%} of job titles translated"))

        df["n_invitations"] = df.groupby("idCv")["is_positive"].transform("sum")
        df["in_eval_set"] = (df["n_invitations"] > 0).astype(int)

        n_eval = df.loc[df["in_eval_set"] == 1, "idCv"].nunique()
        print(f"applicants we can evaluate: {n_eval:,}")

        steps.append(("8. Evaluation set", f"{n_eval:,} applicants with at least one invitation"))

        def region_key(text):
            # "Томская область, Колпашевский район, г Колпашево" -> "томская область"
            text = str(text).split(",")[0].lower().replace("-", " ")
            text = re.sub(r"^(город|г)\s+", "", text)
            return " ".join(text.split())

        # localityName holds region, district and city; regionName holds only the region
        same_region = [region_key(a) == region_key(b) for a, b in zip(df["localityName"], df["regionName"])]
        same_region = np.mean(same_region)

        print(f"applications in the applicant's own region: {same_region:.1%}")
        steps.append(("Region check", f"{same_region:.1%} of applications are in the applicant's own region"))

        KEEP = [
            "idCv", "idVacancy", "cv_status", "is_positive",
            # applicant
            "positionName", "positionName_en", "cv_text", "cv_skills",
            "experience", "salaryMin_cv", "localityName", "n_invitations", "in_eval_set",
            # job
            "vacancyName", "vacancyName_en", "job_text", "job_skills",
            "professionalSphereName", "experienceRequirements", "salaryMin_vacancy",
            "regionName", "datePublished",
        ]

        clean = df[KEEP].sort_values(["idCv", "idVacancy"]).reset_index(drop=True)

        # utf-8-sig lets Excel show the Russian letters correctly
        clean.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")


        print(f"saved {OUT_FILE}")
        print(f"  {len(clean):,} rows x {clean.shape[1]} columns, {os.path.getsize(OUT_FILE) / 1e6:.0f} MB")
        print(f"  applicants {clean['idCv'].nunique():,}   jobs {clean['idVacancy'].nunique():,}")
        print(f"  invitation rate {clean['is_positive'].mean():.1%}")

        report = pd.DataFrame(steps, columns=["Step", "Result"])
        report.to_csv("results/cleaning_report.csv", index=False, encoding="utf-8-sig")
        print("saved results/cleaning_report.csv")
        report
        return Path(OUT_FILE).resolve()
