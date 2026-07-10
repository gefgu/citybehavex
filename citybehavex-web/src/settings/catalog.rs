//! MTUS-based micro-activity catalog, ported verbatim from
//! `citybehavex/activities/catalog.py::_CATALOG_RAW`. Shared by
//! `ActivitiesConfig`'s `durations` validator (Phase 2) and the timeline
//! agent-detail endpoints (Phase 9), which both need the same 25-entry
//! id -> name/description table `web/backend/app/timeline.py` builds once at
//! import time as `_ACTIVITY_BY_ID`.

pub struct ActivityDef {
    pub idx: usize,
    pub name: &'static str,
    pub description: &'static str,
    pub mu_ln: f64,
    pub sigma_ln: f64,
    pub eligible_purposes: &'static [u8],
}

macro_rules! activity {
    ($idx:expr, $name:expr, $desc:expr, $mu:expr, $sigma:expr, [$($p:expr),*]) => {
        ActivityDef {
            idx: $idx,
            name: $name,
            description: $desc,
            mu_ln: $mu,
            sigma_ln: $sigma,
            eligible_purposes: &[$($p),*],
        }
    };
}

/// Same order as `_CATALOG_RAW`; index in this array is the activity id used
/// everywhere else (parquet `activity` columns, PoI/mask arrays, etc).
pub const CATALOG: &[ActivityDef] = &[
    activity!(0, "sleep", "Sleeping or resting at home", 2.08, 0.30, [0]),
    activity!(1, "eatdrink", "Eating, drinking, coffee, lunch, and meal breaks", -0.35, 0.45, [0, 1, 2]),
    activity!(2, "selfcare", "Personal hygiene, grooming, and private care", -0.69, 0.50, [0, 2]),
    activity!(3, "paidwork", "Working at the office or job site", 2.08, 0.30, [1]),
    activity!(4, "educatn", "Studying, attending class, or doing homework", 1.10, 0.50, [2]),
    activity!(5, "foodprep", "Cooking and food preparation", 0.00, 0.60, [0]),
    activity!(6, "cleanetc", "Cleaning, laundry, and other domestic work", 0.00, 0.60, [0]),
    activity!(7, "maintain", "Household maintenance, repairs, and administrative upkeep", 0.00, 0.60, [0]),
    activity!(8, "shopserv", "Shopping and personal services", -0.29, 0.50, [2]),
    activity!(9, "garden", "Gardening and outdoor household work", 0.41, 0.60, [0]),
    activity!(10, "petcare", "Caring for pets and domestic animals", -0.69, 0.50, [0]),
    activity!(11, "eldcare", "Caring for adults or older household members", 0.41, 0.70, [0, 2]),
    activity!(12, "pkidcare", "Physical childcare and supervision", 0.41, 0.70, [0, 2]),
    activity!(13, "ikidcare", "Interactive childcare, play, and homework help", 0.41, 0.70, [0, 2]),
    activity!(14, "religion", "Religious practice, ceremonies, and worship", 0.69, 0.60, [2]),
    activity!(15, "volorgwk", "Volunteering, civic, and organizational work", 0.69, 0.60, [2]),
    activity!(16, "commute", "Commuting to and from work or education", -0.29, 0.50, [1, 2]),
    activity!(17, "travel", "Travel for personal, household, and leisure activities", -0.29, 0.50, [2]),
    activity!(18, "sportex", "Sports, exercise, and gym sessions", 0.41, 0.40, [2]),
    activity!(19, "tvradio", "Watching TV, listening to radio, and passive media", 0.92, 0.60, [0, 2]),
    activity!(20, "read", "Reading books, news, and magazines", 0.41, 0.50, [0, 2]),
    activity!(21, "compint", "Computer, internet, gaming, and online leisure", 0.41, 0.50, [0, 2]),
    activity!(22, "goout", "Going out to restaurants, cinema, theatre, or events", 0.92, 0.50, [2]),
    activity!(23, "leisure", "Social, recreational, and other leisure activities", 0.92, 0.60, [0, 2]),
    activity!(24, "missing", "Unclassified or missing diary time; shown in comparisons only", 0.00, 0.50, []),
];

pub fn by_id(id: i64) -> Option<&'static ActivityDef> {
    CATALOG.get(usize::try_from(id).ok()?)
}

pub fn known_names() -> impl Iterator<Item = &'static str> {
    CATALOG.iter().map(|a| a.name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_has_25_entries_in_index_order() {
        assert_eq!(CATALOG.len(), 25);
        for (i, activity) in CATALOG.iter().enumerate() {
            assert_eq!(activity.idx, i);
        }
        assert_eq!(CATALOG[0].name, "sleep");
        assert_eq!(CATALOG[24].name, "missing");
    }

    #[test]
    fn by_id_matches_index() {
        assert_eq!(by_id(3).unwrap().name, "paidwork");
        assert!(by_id(25).is_none());
        assert!(by_id(-1).is_none());
    }
}
