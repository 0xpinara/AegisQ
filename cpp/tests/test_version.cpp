#include "aegisq/version.hpp"
#include "testing.hpp"

int main() {
    AEGISQ_CHECK(!aegisq::version().empty());
    AEGISQ_CHECK(!aegisq::compiler().empty());
    AEGISQ_CHECK(aegisq::max_threads() >= 1);
    return aegisq::testing::summary("test_version");
}
