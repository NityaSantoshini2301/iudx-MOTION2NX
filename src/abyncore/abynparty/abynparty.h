#ifndef ABYNPARTY_H
#define ABYNPARTY_H

#include <vector>
#include <memory>
#include <omp.h>
#include <fmt/format.h>

#include "utility/constants.h"
#include "utility/abynconfiguration.h"
#include "abynbackend.h"
#include "gate/gate.h"
#include "share/share.h"
//#include "OTExtension/ot/ot-ext.h"


namespace ABYN {

    template<typename T, typename = std::enable_if_t<std::is_unsigned_v<T>>>
    using ArithmeticSharePtr = ABYN::Shares::ArithmeticSharePtr<T>;

    template<typename T, typename = std::enable_if_t<std::is_unsigned_v<T>>>
    using ArithmeticShare = ABYN::Shares::ArithmeticShare<T>;

    class ABYNParty {

    private:
        ABYNConfigurationPtr configuration;
        ABYNBackendPtr backend;

        //Let's make only ABYNConfiguration be copyable
        ABYNParty() = delete;
        ABYNParty(ABYNParty & abynparty) = delete;

    protected:


    public:

        ABYNParty(std::vector<Party> &parties, size_t my_id) {
            configuration = std::make_shared<ABYNConfiguration>(parties, my_id);
            backend = std::make_shared<ABYNBackend>(configuration);
        };

        ABYNParty(std::vector<Party> &&parties, size_t my_id) :
                ABYNParty(parties, my_id) {};

        ABYNParty(std::initializer_list<Party> & list_parties, size_t my_id) {
            configuration = std::make_shared<ABYNConfiguration>(list_parties, my_id);
            backend = std::make_shared<ABYNBackend>(configuration);
        }

        ABYNParty(std::initializer_list<Party> && list_parties, size_t my_id) :
        ABYNParty(list_parties, my_id) {};

        ABYNParty(ABYNConfigurationPtr &configuration) { this->configuration = configuration; };

        ~ABYNParty() {};

        ABYNConfigurationPtr GetConfiguration() { return configuration; };

        template<typename T, typename = std::enable_if_t<std::is_unsigned_v<T>>>
        ArithmeticSharePtr<T> ShareArithmeticInput(bool my_input, T input = 0) {
            auto p = Gates::Arithmetic::ArithmeticInputGate(input, my_input, backend);
            auto s = std::move(p.GetOutputShare());
            auto sa = std::dynamic_pointer_cast<ArithmeticShare<decltype(input)>>(s);
            return sa;
        }

        size_t GetNumOfParties() { return configuration->GetNumOfParties(); };

        void Connect();
    };

    using ABYNPartyPtr = std::unique_ptr<ABYNParty>;

}
#endif //ABYNPARTY_H