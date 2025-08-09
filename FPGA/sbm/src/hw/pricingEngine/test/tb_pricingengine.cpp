

#include <fstream>
#include <iomanip>
#include <iostream>
#include <cstring>
#include <vector>

#include "pricingengine_kernels.hpp"

#define NUM_TEST_SAMPLE_PE (9)

ap_uint<32> float2Uint(float n)
{
    return (ap_uint<32>)(*(ap_uint<32> *)&n);
}

float Uint2Float(ap_uint<32> n)
{
    return (float)(*(float *)&n);
}

int main(int argc, char *argv[])
{

    pricingEngineRegControl_t regControl = {0};
    pricingEngineRegStatus_t regStatus = {0};
    ap_uint<1024> regCapture = 0x0;
    pricingEngineRegStrategy_t regStrategies[NUM_SYMBOL];

    mmInterface intf;
    orderBookResponseVerify_t responseVerify;
    orderBookResponse_t response;
    orderBookResponsePack_t responsePack;
    orderEntryOperation_t operation;
    orderEntryOperationPack_t operationPack;

    orderBookResponseStreamPack_t responseStreamPackFIFO(
        "responseStreamPackFIFO");
    orderEntryOperationStreamPack_t operationStreamPackFIFO(
        "operationStreamPackFIFO");
    clockTickGeneratorEventStream_t eventStreamFIFO("eventStreamFIFO");

    std::cout << "PricingEngine Test" << std::endl;
    std::cout << "------------------" << std::endl;

    memset(&regStrategies, 0, sizeof(regStrategies));


    std::string priceFilePath = "ordBookResp.txt";
    if (argc == 2) priceFilePath = argv[1];
    std::ifstream ifs(priceFilePath.c_str());
    if (!ifs)
    {
        std::cerr << "Error: \"" << priceFilePath << "\" does not exist!!\n";
        return false;
    }



    std::string word;
    ifs >> word;
    while (word == "#")
    {
        std::getline(ifs, word);
        ifs >> word;
    }


    int responseCount{};
    responseCount = std::stoi(word);
#define exchCast(x) (float2Uint(x))


    float bidPrice{};
    float askPrice{};
    std::vector<orderBookResponseVerify_t> orderBookResponses;
    for (int i = 0; i < responseCount; ++i)
    {
        ifs >> bidPrice >> askPrice;
        std::cout << bidPrice << " " << askPrice << "\n";



        orderBookResponses.push_back({i, {1, 0, 0, 0, 0}, {exchCast(1 / bidPrice), 0, 0, 0, 0}, {1, 0, 0, 0, 0}, {1, 0, 0, 0, 0}, {exchCast(askPrice), 0, 0, 0, 0}, {1, 0, 0, 0, 0}});
    }


    for (int i = 0; i < responseCount; ++i)
    {
        responseVerify = orderBookResponses[i];

        response.symbolIndex = responseVerify.symbolIndex;

        response.bidCount =
            (responseVerify.bidCount[4], responseVerify.bidCount[3],
             responseVerify.bidCount[2], responseVerify.bidCount[1],
             responseVerify.bidCount[0]);

        response.bidPrice =
            (responseVerify.bidPrice[4], responseVerify.bidPrice[3],
             responseVerify.bidPrice[2], responseVerify.bidPrice[1],
             responseVerify.bidPrice[0]);

        response.bidQuantity =
            (responseVerify.bidQuantity[4], responseVerify.bidQuantity[3],
             responseVerify.bidQuantity[2], responseVerify.bidQuantity[1],
             responseVerify.bidQuantity[0]);

        response.askCount =
            (responseVerify.askCount[4], responseVerify.askCount[3],
             responseVerify.askCount[2], responseVerify.askCount[1],
             responseVerify.askCount[0]);

        response.askPrice =
            (responseVerify.askPrice[4], responseVerify.askPrice[3],
             responseVerify.askPrice[2], responseVerify.askPrice[1],
             responseVerify.askPrice[0]);

        response.askQuantity =
            (responseVerify.askQuantity[4], responseVerify.askQuantity[3],
             responseVerify.askQuantity[2], responseVerify.askQuantity[1],
             responseVerify.askQuantity[0]);

        intf.orderBookResponsePack(&response, &responsePack);
        responseStreamPackFIFO.write(responsePack);
    }


    regControl.control = 0x12345678;
    regControl.config = 0xdeadbeef;
    regControl.capture = 0x00000000;


    regStrategies[0].select = STRATEGY_PEG;
    regStrategies[0].enable = 0xff;


    regControl.strategy = 0x80000002;


    while (!responseStreamPackFIFO.empty())
    {
        pricingEngineTop(regControl, regStatus, regCapture, regStrategies,
                         responseStreamPackFIFO, operationStreamPackFIFO,
                         eventStreamFIFO);
    }


    while (!operationStreamPackFIFO.empty())
    {
        operationPack = operationStreamPackFIFO.read();
        intf.orderEntryOperationUnpack(&operationPack, &operation);

        std::cout << "ORDER_ENTRY_OPERATION: {" << operation.opCode << ","
                  << operation.symbolIndex << "," << operation.orderId << ","

                  << operation.quantity << "," << reinterpret_cast<float &>(operation.price) << ","
                  << operation.direction << "}" << std::endl;
    }


    std::cout << "--" << std::hex << std::endl;
    std::cout << "STATUS: ";
    std::cout << "PE_STATUS=" << regStatus.status << " ";
    std::cout << "PE_RX_RESP=" << regStatus.rxResponse << " ";
    std::cout << "PE_PROC_RESP=" << regStatus.processResponse << " ";
    std::cout << "PE_TX_OP=" << regStatus.txOperation << " ";

    std::cout << "regERMInitConstr=" << regStatus.strategyNone << " ";
    std::cout << "regSBMExecStatus=" << regStatus.strategyPeg << " ";
    std::cout << "countAncillaFlip=" << regStatus.strategyLimit << " ";
    std::cout << "PE_STRATEGY_NA=" << regStatus.strategyUnknown << " ";
    std::cout << "PE_RX_EVENT=" << regStatus.rxEvent << " ";
    std::cout << "PE_DEBUG=" << regStatus.debug << " ";
    std::cout << std::endl;

    std::cout << std::endl;
    std::cout << "Done!" << std::endl;

    return 0;
}
