#include <stdexcept>

#include "aegisq/distributed_layout.hpp"
#include "testing.hpp"

using aegisq::DistributedLayout;

int main() {
    AEGISQ_CHECK(aegisq::is_power_of_two(8));
    AEGISQ_CHECK(!aegisq::is_power_of_two(6));
    AEGISQ_CHECK(aegisq::log2_exact(16) == 4);

    {  // Default placement puts the highest-numbered qubits on ranks
        DistributedLayout layout(6, 4, 2);
        AEGISQ_CHECK(layout.num_local_qubits() == 4);
        AEGISQ_CHECK(layout.num_global_qubits() == 2);
        AEGISQ_CHECK(layout.local_state_size() == 16);
        AEGISQ_CHECK(layout.is_local(3));
        AEGISQ_CHECK(layout.is_global(4));
        AEGISQ_CHECK(layout.global_position(5) == 1);
        // rank 2 = 0b10: global position 0 is clear, position 1 is set.
        AEGISQ_CHECK(layout.global_bit(4) == 0);
        AEGISQ_CHECK(layout.global_bit(5) == 1);
        AEGISQ_CHECK(layout.partner_rank_for_global_qubit(4) == 3);
        AEGISQ_CHECK(layout.partner_rank_for_global_qubit(5) == 0);
        AEGISQ_CHECK(layout.physical_index(5) == 2 * 16 + 5);
        AEGISQ_CHECK(layout.is_identity_mapping());
    }

    {  // An explicit mapping permutes which qubits are global
        DistributedLayout layout(6, 4, 0, {0, 4, 1, 2, 3, 5});
        AEGISQ_CHECK(!layout.is_identity_mapping());
        const auto globals = layout.global_qubits();
        AEGISQ_CHECK(globals.size() == 2);
        AEGISQ_CHECK(globals[0] == 1);
        AEGISQ_CHECK(globals[1] == 5);
        AEGISQ_CHECK(layout.is_global(1));
        AEGISQ_CHECK(layout.is_local(4));

        // Index translation must be an exact bijection.
        for (std::uint64_t logical = 0; logical < 64; ++logical) {
            const std::uint64_t physical = layout.to_physical_index(logical);
            AEGISQ_CHECK(layout.to_logical_index(physical) == logical);
        }
    }

    {  // Invalid configurations are rejected
        bool threw = false;
        try {
            DistributedLayout(8, 6, 0);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        AEGISQ_CHECK(threw);

        threw = false;
        try {
            DistributedLayout(2, 4, 0);  // no local qubits left
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        AEGISQ_CHECK(threw);

        threw = false;
        try {
            DistributedLayout(4, 2, 0, {0, 0, 1, 2});  // not a permutation
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        AEGISQ_CHECK(threw);
    }

    return aegisq::testing::summary("test_layout");
}
