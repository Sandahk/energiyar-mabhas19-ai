"""Persian display text for the answer composer: field names, unit readings and templates.

Presentation only. No template carries a regulatory value. The only digits in templates name
the corpus itself («مبحث ۱۹», :data:`CORPUS_NUMBERS`); a note template quotes digits only when
the engine note it renders carries them. Every other number a user sees comes from the question,
a certified record or the source metadata (checked by :mod:`mabhas19.generation.provenance`).
"""

from __future__ import annotations

# dimensions and entity kinds of the structured schemas
FIELD_LABELS: dict[str, str] = {
    "occupancy": "کاربری",
    "climate_class": "اقلیم",
    "rating": "رده انرژی",
    "area": "مساحت",
    "material": "مصالح",
    "density": "چگالی",
    "condition": "شرایط",
    "requirement": "الزام",
    "group": "گروه ساختمان",
    "year": "سال",
    "degree_days": "روز-درجه",
    "place": "شهر",
    "province": "استان",
    "wmo_code": "کد ایستگاه هواشناسی",
    "r_ratio": "نسبت R",
    "element": "جزء پوسته ساختمان",
    "component": "بخش پوسته نورگذر",
    "row_condition": "ردیف",
    "equipment": "تجهیزات",
    "place_of_use": "محل مصرف",
    "use": "کاربری",
    "space": "فضا",
}

# the identity kind ``label`` names a different thing in each main table
LABEL_KIND: dict[str, str] = {
    "t19-2-1": "الزام",
    "t19-3-1": "اقلیم",
    "t19-5-1": "جزء پوسته ساختمان",
    "t19-5-2": "بخش پوسته نورگذر",
    "t19-5-3": "تجهیزات",
    "t19-5-4": "تجهیزات",
    "t19-5-5": "محل مصرف",
    "t19-5-6": "کاربری",
    "t19-5-7": "فضا",
}

OUTPUT_LABELS: dict[str, str] = {
    "climate_class": "رده اقلیمی",
    "max_eui": "حداکثر شدت مصرف انرژی مجاز",
    "thermal_conductivity": "ضریب هدایت حرارتی",
    "density": "چگالی",
    "printed_symbol": "علامت درج‌شده در جدول",
    "criterion": "معیار روز-درجه",
    "wmo_code": "کد ایستگاه هواشناسی",
    "r_range": "بازه نسبت R",
    "rating": "رده انرژی",
    "min_r_value": "حداقل مقاومت حرارتی (R)",
    "max_value": "حداکثر مقدار مجاز",
    "threshold": "حداقل بازدهی",
    "energy_rating": "رده انرژی",
    "standard": "استاندارد",
    "max_temperature": "حداکثر دمای آب گرم مصرفی",
    "min_daylight_area": "حداقل نسبت مساحت کفایت نور روز",
    "min_uniformity": "حداقل یکنواختی نور مصنوعی",
    "place": "شهر",
    "occupancy": "کاربری",
}

# printed fields of a row shown as its conditions (other fields are bookkeeping)
CONDITION_LABELS: dict[str, str] = {
    "category": "دسته",
    "subgroup": "زیرگروه",
    "condition": "شرایط",
    "density": "چگالی",
    "climate_class": "اقلیم",
    "climate_group": "اقلیم‌ها",
    "climate_classes": "اقلیم‌ها",
    "energy_rating": "رده انرژی",
    "building_group": "گروه ساختمان",
    "year": "سال",
    "province": "استان",
    "wmo_code": "کد ایستگاه هواشناسی",
    "specification": "مشخصات",
    "standard": "استاندارد",
    "section": "بخش",
    "row": "ردیف",
    "r_range": "بازه نسبت R",
}

UNIT_LABELS: dict[str, str] = {
    "kWh/m2.yr": "کیلووات‌ساعت بر مترمربع در سال",
    "W/m.K": "وات بر متر کلوین",
    "kg/m3": "کیلوگرم بر مترمکعب",
    "m2.K/W": "مترمربع کلوین بر وات",
    "m2.k/W": "مترمربع کلوین بر وات",
    "W/m2.k": "وات بر مترمربع کلوین",
    "W/m2.K": "وات بر مترمربع کلوین",
    "%": "درصد",
    "در صد": "درصد",
    "°C": "درجه سانتی‌گراد",
    "درجه سانتیگراد": "درجه سانتی‌گراد",
}

SYMBOLS: dict[str, str] = {
    "check_mark": "✓ (علامت تیک)",
    "cross_mark": "✗ (علامت ضربدر)",
}

YEAR_ORDINALS: dict[str, str] = {
    "1": "سال اول",
    "2": "سال دوم",
    "3": "سال سوم",
    "4": "سال چهارم",
    "5": "سال پنجم",
}

CORPUS_NUMBERS = frozenset({"19"})  # «مبحث ۱۹» in templates

INTERPRETATION_TITLE = "برداشت سیستم از سؤال شما"
REQUESTED = "مقدار خواسته‌شده"

# fail-closed outcomes of the structured planner (retrieval.plan.OUTCOMES) and route refusals
OUTCOME_MESSAGES: dict[str, str] = {
    "missing_required_dimension": "برای پاسخ دقیق، اطلاعاتی لازم است که در سؤال مشخص نشده است.",
    "unresolved_constraint": (
        "بخشی از سؤال را نمی‌توان بدون حدس به ستون‌ها یا ردیف‌های جدول نسبت داد؛ "
        "برای جلوگیری از پاسخ نادرست، مقداری ارائه نمی‌شود."
    ),
    "invalid_constraint": "مقداری که در سؤال آمده در این جدول چاپ نشده است.",
    "no_printed_category": "در جدول ردیفی که همه شرایط سؤال را داشته باشد چاپ نشده است.",
    "unsupported_reverse_lookup": ("این جدول جستجوی معکوس (از مقدار به ردیف) را پشتیبانی نمی‌کند."),
    "unsupported_output": "مقدار خواسته‌شده در این جدول چاپ نشده است.",
    "unsupported_calculation": (
        "محاسبه (مانند میانگین، درون‌یابی، جمع یا نسبت) انجام نمی‌شود؛ "
        "فقط مقادیر چاپ‌شده در منبع ارائه می‌شود."
    ),
    "unknown_entity": "مورد ذکرشده در فهرست‌های مبحث ۱۹ یافت نشد.",
    "ambiguous_entity": (
        "نام ذکرشده با بیش از یک ردیف چاپ‌شده مطابقت دارد و سؤال مشخص نمی‌کند کدام منظور است."
    ),
    "contradiction": "سؤال شامل شرایط متناقض است.",
    "certification_failed": (
        "نتیجه در بررسی مستقل نهایی تأیید نشد؛ برای ایمنی مقداری ارائه نمی‌شود."
    ),
    "no_result": "برای این سؤال پاسخ قطعی و تأییدشده‌ای در جدول‌های مبحث ۱۹ یافت نشد.",
    "display_check_failed": (
        "نمایش این پاسخ در بررسی ایمنی نهایی رد شد؛ برای ایمنی مقداری نمایش داده نمی‌شود."
    ),
}

# specific reason codes refine the outcome message
REASON_MESSAGES: dict[str, str] = {
    "unknown_place": "شهر ذکرشده در فهرست شهرهای مبحث ۱۹ و پیوست‌های آن یافت نشد.",
    "unknown_material": "این مصالح در جدول مشخصات حرارتی مصالح یافت نشد.",
    "unknown_occupancy": "این کاربری در فهرست کاربری‌های چاپ‌شده یافت نشد.",
    "unknown_label": "مورد ذکرشده در ردیف‌های چاپ‌شده این جدول یافت نشد.",
    "climate_class_not_in_table": "اقلیم ذکرشده در ستون‌های این جدول چاپ نشده است.",
    "rating_not_in_table": "رده ذکرشده در این جدول چاپ نشده است.",
    "unresolved_modifier": (
        "واژه‌هایی در سؤال، مورد خواسته‌شده را توصیف می‌کنند و ممکن است ردیف پاسخ را تغییر "
        "دهند؛ سیستم آن‌ها را بدون حدس تفسیر نمی‌کند."
    ),
    "unattested_quantity": "کمیت خواسته‌شده در این جدول چاپ نشده است.",
    "no_matching_row": "در این جدول ردیفی که همه شرایط سؤال را داشته باشد چاپ نشده است.",
}

MISSING_HINTS: dict[str, str] = {
    "climate_class": "اقلیم را با کد آن یا با نام شهر مشخص کنید.",
    "rating": "رده انرژی ساختمان را مشخص کنید.",
    "occupancy": "کاربری ساختمان را دقیق‌تر بنویسید.",
    "group": "گروه ساختمان را مشخص کنید.",
    "year": "سال اجرا را مشخص کنید.",
    "density": "چگالی مصالح را مشخص کنید.",
    "area": "مساحت زیربنای ساختمان را مشخص کنید.",
    "material": "نام مصالح را مشخص کنید.",
    "place": "نام شهر را مشخص کنید.",
    "element": "جزء پوسته را مشخص کنید (مثلاً نوع دیوار، سقف، کف یا در).",
    "component": "بخش پوسته نورگذر را مشخص کنید (مثلاً نوع پنجره).",
    "equipment": "نوع تجهیز را مشخص کنید.",
    "place_of_use": "محل مصرف آب گرم را مشخص کنید.",
    "use": "کاربری را مشخص کنید.",
    "space": "نوع فضا را مشخص کنید.",
    "requirement": "الزام مورد نظر جدول را مشخص کنید.",
}

AMBIGUOUS_HINT = "لطفاً یکی از گزینه‌های چاپ‌شده را دقیق در سؤال بنویسید."

GATE_MESSAGES: dict[str, str] = {
    "commercial": (
        "قیمت، بازار و موضوعات تجاری در مبحث ۱۹ مطرح نشده‌اند و از این منبع پاسخ داده نمی‌شوند."
    ),
    "service_information": (
        "اطلاعات خدماتی (مانند نشانی، تماس یا معرفی شرکت‌ها) در مبحث ۱۹ وجود ندارد."
    ),
    "education_services": "اطلاعات دوره‌ها و خدمات آموزشی در مبحث ۱۹ وجود ندارد.",
    "outside_mabhas19": "این موضوع در محدوده مبحث ۱۹ (صرفه‌جویی در مصرف انرژی) قرار ندارد.",
}
OUT_OF_SCOPE = "پاسخ این سؤال در متن و جدول‌های بارگذاری‌شده مبحث ۱۹ یافت نشد."
OUT_OF_SCOPE_REFERENCE = "بند یا جدول ذکرشده در ویرایش پنجم مبحث ۱۹ وجود ندارد."
COVERED_BY = "مرجع مرتبط:"

HEADLINE_PROSE = "بخش‌های مرتبط از متن مبحث ۱۹:"
HEADLINE_ROWS = "ردیف‌های چاپ‌شده مطابق سؤال شما:"
HEADLINE_REVERSE = "ردیف‌های چاپ‌شده‌ای که مقدار ذکرشده در سؤال را دارند:"
HEADLINE_DIFFERENT = "منابع برای این مورد مقادیر متفاوتی چاپ کرده‌اند؛ هر دو نمایش داده شده است."

CONFLICT = (
    "منابع درباره این مورد با هم اختلاف دارند؛ هر دو مقدار نمایش داده شده و هیچ‌کدام ترجیح "
    "داده نشده است."
)
SOURCE_ISSUE = "مورد ثبت‌شده درباره منبع"

# engine notes with a fixed wording, shown in Persian; any other note stays technical
NOTE_PREFIXES: tuple[tuple[str, str], ...] = (
    (
        "Appendix 3 has no 5C column",
        "پیوست ۳ ستونی برای اقلیم 5C ندارد؛ برای این اقلیم حداکثر شدت مصرف انرژی چاپ نشده "
        "و از ستون دیگری هم برداشته نمی‌شود.",
    ),
    (
        "interpolation, extrapolation, averages",
        "درون‌یابی، برون‌یابی، میانگین، جمع و نسبت انجام نمی‌شود؛ فقط مقادیر چاپ‌شده ارائه می‌شود.",
    ),
    (
        "density equals a printed bound",
        "چگالی ذکرشده دقیقاً روی مرز دو بازه چاپ‌شده است و منبع مشخص نکرده مرز جزو کدام بازه "
        "است؛ هر دو ردیف نمایش داده شده است.",
    ),
    (
        "enumeration of printed alternatives",
        "همه مقادیر چاپ‌شده برای این مورد نمایش داده شده است؛ هر ردیف شرایط خودش را دارد.",
    ),
)
NOTE_ALTERNATIVE = "برای این ردیف، به‌جای مقدار خواسته‌شده، خروجی جایگزین اعلام‌شده جدول چاپ شده است."

PILOT_WARNING = (
    "این دستیار آزمایشی مبحث ۱۹ است. برای تصمیم‌های مهم مهندسی، پاسخ را با منبع "
    "استنادشده تطبیق دهید."
)
