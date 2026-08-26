"""
LEGACY reference script - NOT used by the live interview drive.

Until 2026-08-20 tests/e2e/test_bot_responsiveness.py spoke these
ten fixed answers regardless of the interviewer's question. The
live drive now answers the ACTUAL interviewer question through the
CandidateSimulator (simulator/live_answers.py). This list is kept
ONLY as the fixture for the offline proofs that scripted text can
never reach a judge or be produced by the simulator:
  tests/evaluation/test_interviewer_turn_evaluation.py
  tests/simulator/test_candidate_simulator_offline.py
"""

CANDIDATE_ANSWERS = [
    (
        "Hello Jamie, thank you, it's nice to meet you. I'm Alex, "
        "a software engineer with about six years of experience, "
        "mostly on backend systems in Python and Node. I started "
        "out at a logistics startup where I built order-tracking "
        "APIs, and for the last three years I've been at a "
        "payments company where I own the transaction-processing "
        "services. Along the way I've picked up a lot of "
        "operational experience too, things like on-call, "
        "observability and capacity planning, and I genuinely "
        "enjoy the reliability side of engineering as much as the "
        "feature side."
    ),
    (
        "The project I'm proudest of recently is a migration of "
        "our payments monolith to microservices. We had a single "
        "Django application handling everything from checkout to "
        "reconciliation, and deployments had become genuinely "
        "risky, a bad release could block refunds for hours. I "
        "led the effort to carve out the highest-risk domains "
        "first, starting with the ledger service. We used the "
        "strangler pattern, kept a compatibility layer so the "
        "monolith and the new services could co-exist, and moved "
        "traffic over gradually with feature flags. After about "
        "nine months our deployment frequency went from weekly "
        "to daily and rollbacks became a five-minute operation "
        "instead of an incident."
    ),
    (
        "I'd say my biggest strength is debugging complex "
        "distributed problems calmly. A good example is a "
        "duplicate-charge bug we saw roughly once in every "
        "hundred thousand transactions. It turned out to be a "
        "retry storm caused by a load balancer timeout that was "
        "shorter than our downstream payment provider's p99 "
        "latency, so a small percentage of requests were retried "
        "while the original was still in flight. I found it by "
        "correlating trace IDs across three services and then "
        "proved it with a reproduction in staging. My other "
        "strength is mentoring, I've onboarded four junior "
        "engineers and I really enjoy code review as a teaching "
        "tool rather than a gate."
    ),
    (
        "The most challenging system I've built was a real-time "
        "analytics pipeline processing around forty million "
        "events a day. The hard part wasn't throughput, it was "
        "correctness under failure. We used Kafka with "
        "exactly-once semantics into a stream processor, and we "
        "had to design idempotent consumers because upstream "
        "producers could still replay events after network "
        "partitions. I also had to make late-arriving data "
        "visible without corrupting already-published daily "
        "aggregates, which we solved with watermarking and a "
        "correction topic. It taught me a lot about the "
        "difference between a system that works in the demo and "
        "one that survives a bad week in production."
    ),
    (
        "When I approach a new problem I try to resist jumping "
        "straight to a solution. I usually start by writing down "
        "what I actually know versus what I'm assuming, and then "
        "I look for the cheapest experiment that can invalidate "
        "the riskiest assumption. For example, before we "
        "committed to sharding our main database, I spent two "
        "days proving with production query logs that eighty "
        "percent of our load came from three query patterns that "
        "a read replica plus better indexing could absorb. That "
        "saved us a quarter of very painful migration work. I "
        "also like to write a one-page design doc even for "
        "medium-sized changes, mostly because the act of writing "
        "exposes the gaps in my thinking."
    ),
    (
        "On collaboration, I work most closely with product "
        "managers, designers and our SRE team. The habit that "
        "has served me best is over-communicating state: I keep "
        "a running thread for every project with what shipped, "
        "what's blocked and what decision I need next. With "
        "product specifically I try to translate technical "
        "trade-offs into user-visible terms, so instead of "
        "saying a queue adds eventual consistency, I'll say the "
        "receipt email may arrive up to a minute late, is that "
        "acceptable. It makes prioritisation conversations much "
        "faster and less adversarial."
    ),
    (
        "In five years I'd like to be leading a small team, "
        "maybe five or six engineers, while still writing code "
        "myself, probably around a third of my time. I've had a "
        "taste of that as a tech lead on the migration project "
        "and I found I enjoy the multiplier effect, unblocking "
        "people, shaping the architecture, and doing the "
        "hands-on work on the gnarliest pieces. I'm deliberately "
        "not aiming at pure people management yet, I think I "
        "have a few more years of deep technical growth in me "
        "first, especially around large-scale data systems."
    ),
    (
        "Disagreements happen a lot in engineering and I think "
        "that's healthy. My approach is to first make sure I can "
        "state the other person's position well enough that they "
        "would agree with my summary, because half the time the "
        "disagreement dissolves right there. When it doesn't, I "
        "try to convert opinions into testable claims. We had a "
        "long argument about GraphQL versus REST for a partner "
        "API, and we settled it by prototyping both against the "
        "three most complex partner use cases and measuring "
        "integration effort. REST won for our case, and the "
        "person who had championed GraphQL wrote the decision "
        "record, which kept the outcome blameless."
    ),
    (
        "What motivates me most is seeing software I built get "
        "used in the real world at meaningful scale. At the "
        "payments company there's a dashboard showing live "
        "transaction volume, and knowing my ledger service sits "
        "under every one of those transactions is genuinely "
        "satisfying. I'm also motivated by craft, I like leaving "
        "a codebase measurably better than I found it, whether "
        "that's cutting a flaky test suite's runtime in half or "
        "deleting a thousand lines of dead configuration. And "
        "honestly, I like working with people who care, energy "
        "is contagious in both directions."
    ),
    (
        "I think that covers my background well. To summarise, "
        "six years of backend engineering, deep experience with "
        "the microservices migration and the analytics pipeline "
        "I described, strengths in distributed debugging and "
        "mentoring, and I'm looking for a role where I can grow "
        "into technical leadership. Thank you for the "
        "conversation, Jamie, I've enjoyed the questions. I "
        "don't have any further questions from my side."
    ),
]
