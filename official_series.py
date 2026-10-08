"""Official state-wise pending cases in District & Subordinate Courts at year end, 2014-2025.

Sources (public answers in Parliament, Ministry of Law & Justice; figures from NJDG / Supreme Court of India):
  2014-2022  Lok Sabha Unstarred Q. 1838, 16.12.2022, Annexure-III
             https://sansad.in/getFile/loksabhaquestions/annex/1710/AU1838.pdf?source=pqals
  2023-2025  Lok Sabha Unstarred Q. 2362, 13.02.2026, Annexure-II
             https://sansad.in/getFile/loksabhaquestions/annex/187/AU2362_8Lccb3.pdf?source=pqals
Units that split or merged during the period are combined so every row is one consistent unit:
Andhra Pradesh + Telangana, Jammu & Kashmir + Ladakh, Dadra & Nagar Haveli + Daman & Diu.

Known issues in the sources (kept visible, never silently patched):
  - The "2022" pending column of USQ 1838 is "till date" in an answer dated 16.12.2022, so it is a mid-December
    figure, not 31.12.2022. Growth into 2022 is slightly understated and growth into 2023 slightly overstated.
  - Disposals (USQ 1838, Annexure-VIII) for 2022 cover only part of the year and are excluded.
  - Kerala's 2016 disposals are printed as 11,993,996, about ten times every neighbouring year; the printed national
    total (29,917,126) includes it. It is set to missing here rather than guessed.
"""
from __future__ import annotations

from typing import Final

import pandas as pd

YEARS: Final = tuple(range(2014, 2026))
SOURCES: Final = (
    "Lok Sabha USQ 1838 (16.12.2022), Annexure-III: 2014-2022",
    "Lok Sabha USQ 2362 (13.02.2026), Annexure-II: 2023-2025",
)
_2014_2022: Final = {
    "Uttar Pradesh": (5517004, 5574490, 5980071, 6390684, 6987417, 7807863, 8781104, 9966606, 10641073),
    "Andhra Pradesh + Telangana": (1014372, 1031515, 1077944, 1040864, 1068400, 567096 + 580193, 649157 + 691646,
                                   785379 + 790360, 827790 + 822658),
    "Maharashtra": (2868764, 2994074, 3239540, 3340050, 3531425, 3821487, 4504573, 4800895, 4919254),
    "Goa": (35001, 39615, 42074, 39249, 42783, 49049, 58967, 59414, 56082),
    "Dadra & Nagar Haveli + Daman & Diu": (4717, 5626, 5486, 5295, 5468, 5344, 6281, 6523, 2857 + 3784),
    "West Bengal": (2556461, 2618813, 2728753, 2141254, 1950492, 2048697, 2170788, 2384020, 2481419),
    "Andaman & Nicobar": (9230, 9495, 8767, 9227, 10229, 9795, 9839, 9321, 9163),
    "Chhattisgarh": (278887, 285962, 290434, 277338, 267429, 285025, 331849, 381984, 403266),
    "Delhi": (459267, 568909, 636121, 747704, 834813, 882366, 1018642, 1231373, 1440149),
    "Gujarat": (2179979, 2142011, 1822311, 1555203, 1447459, 1595813, 1917992, 1952262, 1808627),
    "Assam": (240597, 242503, 258639, 276520, 291960, 301427, 360753, 415024, 478356),
    "Nagaland": (3553, 3862, 4430, 4749, 4994, 3361, 4206, 4569, 4605),
    "Meghalaya": (14249, 14988, 15239, 14775, 13584, 13673, 15830, 16010, 15576),
    "Manipur": (15147, 6885, 6978, 6799, 6216, 6516, 6957, 8183, 7654),
    "Tripura": (115209, 129789, 148275, 107089, 58261, 27491, 44654, 43096, 38986),
    "Mizoram": (3730, 4671, 4665, 5148, 6154, 6589, 6338, 6304, 5843),
    "Arunachal Pradesh": (5895, 8776, 14583, 9878, 9652, 10658, 12651, 14318, 16029),
    "Himachal Pradesh": (226224, 206727, 235193, 234639, 256640, 293706, 420891, 464892, 504912),
    "Jammu & Kashmir + Ladakh": (185078, 199699, 145999, 161674, 163520, 172769, 198771, 216245, 258228),
    "Jharkhand": (315484, 324357, 342768, 338680, 330607, 365642, 427130, 490905, 499687),
    "Karnataka": (1226112, 1268966, 1362167, 1432952, 1494608, 1531008, 1709220, 1780802, 1878045),
    "Kerala": (1331558, 1345127, 1482667, 1623212, 1652509, 1614277, 2089289, 2089147, 1992343),
    "Lakshadweep": (418, 380, 357, 354, 364, 397, 453, 470, 539),
    "Madhya Pradesh": (1181459, 1191799, 1260637, 1332566, 1354602, 1455435, 1727293, 1920613, 1957175),
    "Tamil Nadu": (1038820, 1082793, 1071366, 1065878, 1084286, 1137684, 1263758, 1331944, 1383865),
    "Puducherry": (24431, 24973, 28155, 26930, 27161, 30094, 33470, 32998, 32216),
    "Odisha": (1070377, 1064039, 1049325, 1178882, 1319031, 1433522, 1592250, 1789677, 1846520),
    "Bihar": (1923649, 2073303, 2128325, 2223744, 2502204, 2714344, 3016743, 3276696, 3434130),
    "Punjab": (507663, 504028, 504320, 572802, 602014, 642327, 843791, 945609, 952777),
    "Haryana": (493768, 524281, 547736, 643394, 728097, 853375, 1101330, 1313881, 1445775),
    "Chandigarh": (40414, 36322, 38907, 41695, 56357, 62955, 70633, 72384, 88805),
    "Rajasthan": (1454566, 1479173, 1573986, 1635389, 1732308, 1769823, 1947688, 2162774, 2248201),
    "Sikkim": (999, 1460, 1434, 1405, 1208, 1142, 1455, 1616, 1645),
    "Uttarakhand": (145326, 166618, 190948, 210018, 232338, 195281, 249350, 287204, 318743),
}
_2023_2025: Final = {
    "Andaman & Nicobar": (9070, 8514, 8329), "Andhra Pradesh + Telangana": (895282 + 920101, 921948 + 947417, 915398 + 976399),
    "Arunachal Pradesh": (10671, 9784, 10665), "Assam": (451138, 496819, 564945), "Bihar": (3608014, 3660802, 3700012),
    "Chandigarh": (91078, 104194, 100498), "Chhattisgarh": (418688, 420661, 452049), "Delhi": (1229806, 1452717, 1587493),
    "Goa": (63159, 60895, 61285), "Gujarat": (1556371, 1503763, 1590844), "Haryana": (1524118, 1446433, 1521463),
    "Himachal Pradesh": (593875, 646753, 590988), "Jammu & Kashmir + Ladakh": (310486 + 1244, 311925 + 1407, 345785 + 1583),
    "Jharkhand": (560102, 547977, 564410), "Karnataka": (1987983, 2113569, 2237391), "Kerala": (1897469, 1783932, 1788680),
    "Lakshadweep": (492, 518, 539), "Madhya Pradesh": (2055620, 2054704, 2098396),
    "Maharashtra": (5326823, 5612876, 5926999), "Manipur": (13286, 12857, 13931), "Meghalaya": (16068, 15178, 16343),
    "Mizoram": (3983, 6298, 6875), "Nagaland": (3201, 3357, 3856), "Odisha": (1687827, 1741306, 1793888),
    "Puducherry": (37477, 35381, 36495), "Punjab": (876134, 864524, 914711), "Rajasthan": (2525123, 2496501, 2542253),
    "Sikkim": (1819, 1727, 1962), "Tamil Nadu": (1508744, 1520258, 1735167),
    "Dadra & Nagar Haveli + Daman & Diu": (7314, 7750, 8346), "Tripura": (44490, 44085, 58295),
    "Uttar Pradesh": (11444974, 11648631, 11345328), "Uttarakhand": (350474, 350069, 300614),
    "West Bengal": (2996527, 3380587, 3835113),
}
# cases disposed during each year, USQ 1838 Annexure-VIII (2022 is a partial year and is left out)
_DISPOSED_2014_2021: Final = {
    "Uttar Pradesh": (3182318, 3313424, 3618460, 3288866, 3282885, 3426942, 2274687, 3972255),
    "Andhra Pradesh + Telangana": (647130, 658713, 603017, 760582, 741390, 364947 + 331963, 166918 + 133518,
                                   244105 + 368092),
    "Maharashtra": (1536322, 1649187, 2281027, 2378096, 2196271, 1877895, 752986, 1388604),
    "Goa": (30625, 34765, 34130, 34814, 36235, 32634, 14130, 32953),
    "Dadra & Nagar Haveli + Daman & Diu": (2771, 3323, 3810, 3302, 4001, 4081, 2225, 3875),
    "West Bengal": (1078273, 1091807, 1050880, 1694427, 1016319, 683238, 307850, 476809),
    "Andaman & Nicobar": (11036, 7936, 8761, 7776, 7284, 8563, 4054, 10124),
    "Chhattisgarh": (176144, 195174, 195514, 208498, 229548, 214399, 78278, 195240),
    "Delhi": (930732, 636078, 644624, 740779, 808156, 814555, 245879, 353683),
    "Gujarat": (1132433, 1093664, 1586926, 1386529, 1418688, 1142383, 394455, 1447320),
    "Assam": (276138, 272538, 251119, 313617, 311150, 254823, 94574, 182346),
    "Nagaland": (3047, 4826, 4415, 2957, 3514, 5728, 2488, 3921),
    "Meghalaya": (11691, 18429, 11100, 12316, 8517, 7890, 3163, 5232),
    "Manipur": (14257, 7395, 6588, 5256, 4379, 3717, 1747, 1411),
    "Tripura": (193003, 209282, 185283, 169763, 139931, 90786, 26095, 55417),
    "Mizoram": (10747, 10355, 10905, 12497, 12563, 15107, 11524, 11236),
    "Arunachal Pradesh": (7615, 5238, 4384, 12165, 7499, 7735, 4144, 8156),
    "Himachal Pradesh": (409732, 316717, 322008, 317251, 343667, 483869, 187035, 384726),
    "Jammu & Kashmir + Ladakh": (297507, 392819, 98638, 110825, 146194, 81520, 62465, 109071),
    "Jharkhand": (110068, 118845, 104284, 157765, 194200, 187370, 108247, 142674),
    "Karnataka": (1367041, 1209127, 1079586, 1144693, 1120397, 1272673, 961619, 1848768),
    "Kerala": (1355926, 1338443, None, 983409, 961840, 1005350, 365958, 816047),  # 2016 printed as 11,993,996
    "Lakshadweep": (114, 280, 269, 191, 237, 201, 238, 284),
    "Madhya Pradesh": (1113382, 1073584, 1074131, 1218909, 1386280, 1207541, 681333, 1122497),
    "Tamil Nadu": (1949061, 1151349, 1017111, 1015322, 906184, 849240, 429767, 646592),
    "Puducherry": (33519, 20409, 16624, 16770, 14052, 12137, 6533, 14628),
    "Odisha": (470085, 408261, 468395, 365602, 255005, 296535, 126077, 223485),
    "Bihar": (305570, 292678, 344683, 344981, 361063, 405347, 174478, 354099),
    "Punjab": (549300, 578681, 605324, 718292, 712529, 670175, 333826, 582027),
    "Haryana": (587384, 542440, 593132, 579631, 628939, 614384, 281734, 558068),
    "Chandigarh": (180616, 145990, 143520, 101617, 139172, 146256, 35294, 55242),
    "Rajasthan": (1132028, 1371762, 1378527, 1514181, 1468290, 1508232, 786604, 1192950),
    "Sikkim": (2008, 3806, 550, 2583, 2440, 1906, 987, 1807),
    "Uttarakhand": (220660, 200931, 175405, 237197, 288999, 341452, 143974, 214860),
}
PRINTED_DISPOSED_TOTALS: Final = {2014: 19328283, 2015: 18378256, 2016: 29917126, 2017: 19861459, 2018: 19157818,
                                  2019: 18371574, 2020: 9204884, 2021: 17028604}
KERALA_2016_AS_PRINTED: Final = 11993996

# national totals as printed, for checking the transcription
PRINTED_TOTALS: Final = {2014: 26488408, 2015: 27176029, 2016: 28248600, 2017: 28696040, 2018: 30074590,
                         2019: 32296224, 2020: 37285742, 2021: 41053498, 2022: 42826777,
                         2023: 45029031, 2024: 46236117, 2025: 47657328}


def pending() -> pd.DataFrame:
    """States x years (2014-2025) of pending cases at 31 December."""
    rows = {s: list(a) + list(_2023_2025[s]) for s, a in _2014_2022.items()}
    return pd.DataFrame.from_dict(rows, orient="index", columns=list(YEARS)).sort_index()


def disposed() -> pd.DataFrame:
    """States x years (2014-2021) of cases disposed during the year; Kerala 2016 is missing (see module notes)."""
    return pd.DataFrame.from_dict(_DISPOSED_2014_2021, orient="index", columns=list(range(2014, 2022)),
                                  dtype="float64").sort_index()


def instituted() -> pd.DataFrame:
    """Implied filings by the stock-flow identity: filed(t) = pending(t) - pending(t-1) + disposed(t), 2015-2021.
    Transfers, restorations and data clean-ups also land here, so treat single state-years with care."""
    p, d = pending(), disposed()
    years = list(range(2015, 2022))
    return pd.DataFrame({y: p[y] - p[y - 1] + d[y] for y in years})


def check_totals(tol: float = 0.002) -> dict[int, float]:
    """Relative gap between the summed states and the printed national total, per year."""
    tot = pending().sum()
    return {y: float(tot[y] / PRINTED_TOTALS[y] - 1.0) for y in YEARS if abs(tot[y] / PRINTED_TOTALS[y] - 1.0) > tol}
