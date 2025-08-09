

#include <cstring>

#include "unistd.h"
#include "xlnx_cmd_manager.h"
#include "xlnx_shell.h"
#include "xlnx_shell_cmds.h"



using namespace XLNX;



static const char* HELP_STRING  = "help";
static const char* ALL_STRING   = "all";
static const char* LINE_STRING  = "----------------------------------------------------------------------------------";


CommandManager::CommandManager()
{

	memset(&m_builtInCommandTableDescriptor, 0, sizeof(m_builtInCommandTableDescriptor));


	memset(m_objectCommandTableDescriptors, 0, sizeof(m_objectCommandTableDescriptors));
	m_numObjectCommandTables = 0;

    ResetCompletion();
}









CommandManager::~CommandManager()
{

}







void CommandManager::SetBuiltInCommandTable(CommandTableElement* pTable, uint32_t tableLength)
{
	m_builtInCommandTableDescriptor.pObjectData = nullptr;
	m_builtInCommandTableDescriptor.pObjectName = nullptr;
	m_builtInCommandTableDescriptor.pTable		= pTable;
	m_builtInCommandTableDescriptor.tableLength = tableLength;
}








bool CommandManager::AddObjectCommandTable(const char* objectName, void* pObjectData, CommandTableElement* pTable, uint32_t tableLength)
{
	bool bOKToContinue = true;
	int i;
	ObjectCommandTableDescriptor* pDescriptor;


	if (m_numObjectCommandTables >= MAX_OBJECT_COMMAND_TABLES)
	{

		bOKToContinue = false;
	}



	if (bOKToContinue)
	{
		for (i = 0; i < (int)m_numObjectCommandTables; i++)
		{
			pDescriptor = &m_objectCommandTableDescriptors[i];

			if (strcmp(pDescriptor->pObjectName, objectName) == 0)
			{

				bOKToContinue = false;
				break;
			}
		}
	}





	if (bOKToContinue)
	{


		pDescriptor = &m_objectCommandTableDescriptors[m_numObjectCommandTables];

		pDescriptor->pObjectName	= objectName;
		pDescriptor->pObjectData	= pObjectData;
		pDescriptor->pTable			= pTable;
		pDescriptor->tableLength	= tableLength;

		m_numObjectCommandTables++;
	}





	return bOKToContinue;
}








bool CommandManager::ExecuteCommand(Shell* pShell, int argc, char* argv[])
{
	uint32_t i;
	CommandTableElement* pElement;
	bool bFoundCommand = false;
    int commandRetVal = 0;
    bool bCommandOK = false;




    if (strcmp(HELP_STRING, argv[0]) == 0)
    {
        bFoundCommand = true;
        PrintBuiltInCommandHelp(pShell);
        pShell->printf("\n");


        if (argc > 1)
        {
            if (strcmp(ALL_STRING, argv[1]) == 0)
            {
                PrintAllExternalObjectsHelp(pShell);
            }
        }
        else
        {
            PrintExternalObjectNamesOnly(pShell);
        }


    }
    else if (strcmp("date", argv[0]) == 0)
    {
            time_t t = time(NULL);
            struct tm *tm = localtime(&t);
            printf("%s", asctime(tm));
    }
    else if (strcmp("sleep", argv[0]) == 0)
    {
	    if (argc > 1)
            {
		    bool validNum = true;
                    for(int i = 0; i < strlen(argv[1]); i ++)
                    {
			    if (argv[1][i] < 48 || argv[1][i] > 57)
				    validNum = false;

                    }

                    if (validNum)
                    {
			    long arg = strtol(argv[1], NULL, 10);
                            printf("Sleeping for %s second\n", argv[1]);
                            sleep(arg);
                    }
            }
    }
    else
    {

    }




    if (!bFoundCommand)
    {



        for (i = 0; i < m_builtInCommandTableDescriptor.tableLength; i++)
        {
            pElement = &m_builtInCommandTableDescriptor.pTable[i];

            if (strcmp(pElement->commandName, argv[0]) == 0)
            {
                bFoundCommand = true;
                commandRetVal = pElement->handler(pShell, argc, argv, nullptr);
                if (commandRetVal == XLNX_OK)
                {
                    bCommandOK = true;
                }
                break;
            }
        }
    }




	if (!bFoundCommand)
	{





		CommandManager::ObjectCommandTableDescriptor* pObjectCommandTableDescriptor;
		CommandTableElement* pTable;
		uint32_t tableLength;
		bool bFoundDescriptor;

		bFoundDescriptor = GetObjectDescriptor(argv[0], &pObjectCommandTableDescriptor);

		if (bFoundDescriptor)
        {
            if (argc >= 2)
            {


                if (strcmp(argv[1], HELP_STRING) == 0)
                {
                    bFoundCommand = true;
                    PrintSingleExternalObjectHelp(pShell, pObjectCommandTableDescriptor);
                }



                if (!bFoundCommand)
                {
                    pTable = pObjectCommandTableDescriptor->pTable;
                    tableLength = pObjectCommandTableDescriptor->tableLength;

                    for (i = 0; i < tableLength; i++)
                    {
                        pElement = &pTable[i];

                        if (strcmp(pElement->commandName, argv[1]) == 0)
                        {
                            bFoundCommand = true;
                            commandRetVal = pElement->handler(pShell, (argc - 1), &argv[1], pObjectCommandTableDescriptor->pObjectData);
                            if (commandRetVal == XLNX_OK)
                            {
                                bCommandOK = true;
                            }
                            break;
                        }
                    }
                }

				if (!bFoundCommand)
				{
					pShell->printf("Unknown object command: '%s %s'\n", argv[0], argv[1]);
				}
			}
			else
			{


                bFoundCommand = true;
                PrintSingleExternalObjectHelp(pShell, pObjectCommandTableDescriptor);
			}
		}
		else
		{
			if (strcmp("date", argv[0]) == 0 || strcmp("sleep", argv[0]) == 0)
			{

			}
			else
			{
				pShell->printf("Unknown command or object: '%s'\n", argv[0]);
			}
		}
	}


	return bCommandOK;
}











void CommandManager::GetNumObjects(uint32_t* pNumObjects)
{
	*pNumObjects = m_numObjectCommandTables;
}



bool CommandManager::GetObjectDescriptor(uint32_t objectIndex, ObjectCommandTableDescriptor** ppDescriptor)
{
	bool bOKToContinue = true;

	if (objectIndex < m_numObjectCommandTables)
	{
		*ppDescriptor = &m_objectCommandTableDescriptors[objectIndex];
	}
	else
	{
		bOKToContinue = false;
	}

	return bOKToContinue;
}






bool CommandManager::GetObjectDescriptor(char* objectName, ObjectCommandTableDescriptor** ppDescriptor)
{
	bool bFoundMatch = false;
	uint32_t i;
	ObjectCommandTableDescriptor* pDescriptor;



	for (i = 0; i < m_numObjectCommandTables; i++)
	{
		pDescriptor = &m_objectCommandTableDescriptors[i];

		if (strcmp(pDescriptor->pObjectName, objectName) == 0)
		{
			bFoundMatch = true;

			*ppDescriptor = pDescriptor;

			break;
		}
	}


	return bFoundMatch;
}







void CommandManager::PrintBuiltInCommandHelp(Shell* pShell)
{
    CommandTableElement* pElement;

    pShell->printf("+-%.15s-+-%.25s-+-%.40s-+\n", LINE_STRING, LINE_STRING, LINE_STRING);
    pShell->printf("| %-15s | %-25s | %-40s |\n", "Built-in Cmds", "Argument List", "Description");
    pShell->printf("+-%.15s-+-%.25s-+-%.40s-+\n", LINE_STRING, LINE_STRING, LINE_STRING);

    for (uint32_t i = 0; i < XLNX::NUM_SHELL_COMMANDS; i++)
    {
        pElement = &XLNX_SHELL_COMMANDS_TABLE[i];

        pShell->printf("| %-15s | %-25s | %-40s |\n", pElement->commandName, pElement->argsListString, pElement->descriptionString);
    }

    pShell->printf("+-%.15s-+-%.25s-+-%.40s-+\n", LINE_STRING, LINE_STRING, LINE_STRING);
}




void CommandManager::PrintExternalObjectHelpTableHeader(Shell* pShell)
{
    pShell->printf("\n");
    pShell->printf("+-%.15s-+-%.20s-+-%.40s-+-%.45s-+\n", LINE_STRING, LINE_STRING, LINE_STRING, LINE_STRING);
    pShell->printf("| %-15s | %-20s | %-40s | %-45s |\n", "Ext. Object", "Cmds", "Argument List", "Description");
    pShell->printf("+-%.15s-+-%.20s-+-%.40s-+-%.45s-+\n", LINE_STRING, LINE_STRING, LINE_STRING, LINE_STRING);
}







void CommandManager::PrintExternalObjectHelpTableFooter(Shell* pShell)
{
    pShell->printf("+-%.15s-+-%.20s-+-%.40s-+-%.45s-+\n", LINE_STRING, LINE_STRING, LINE_STRING, LINE_STRING);
}







void CommandManager::PrintSingleExternalObjectHelp(Shell* pShell, ObjectCommandTableDescriptor* pDescriptor)
{
    PrintExternalObjectHelpTableHeader(pShell);

    PrintExternalObjectHelpTableDataRows(pShell, pDescriptor);

    PrintExternalObjectHelpTableFooter(pShell);
}




void CommandManager::PrintAllExternalObjectsHelp(Shell* pShell)
{
    uint32_t numObjectCommandTables;
    bool bOKToContinue = true;
    ObjectCommandTableDescriptor* pDescriptor;

    GetNumObjects(&numObjectCommandTables);


    PrintExternalObjectHelpTableHeader(pShell);

    for (uint32_t i = 0; i < numObjectCommandTables; i++)
    {
        bOKToContinue = GetObjectDescriptor(i, &pDescriptor);

        if (bOKToContinue)
        {
            PrintExternalObjectHelpTableDataRows(pShell, pDescriptor);

            PrintExternalObjectHelpTableFooter(pShell);
        }
    }
}





void CommandManager::PrintExternalObjectHelpTableDataRows(Shell* pShell, ObjectCommandTableDescriptor* pDescriptor)
{
    char* pObjectName;
    CommandTableElement* pTable;
    uint32_t tableLength;
    CommandTableElement* pElement;


    pObjectName = (char*)pDescriptor->pObjectName;
    pTable = pDescriptor->pTable;
    tableLength = pDescriptor->tableLength;

    for (uint32_t j = 0; j < tableLength; j++)
    {
        pElement = &pTable[j];

        if (j == 0)
        {
            pShell->printf("| %-15s | %-20s | %-40s | %-45s |\n", pObjectName, pElement->commandName, pElement->argsListString, pElement->descriptionString);
        }
        else
        {
            pShell->printf("| %-15s | %-20s | %-40s | %-45s |\n", "", pElement->commandName, pElement->argsListString, pElement->descriptionString);
        }
    }

}








void CommandManager::PrintExternalObjectNamesOnlyHeader(Shell* pShell)
{
    pShell->printf("\n");
    pShell->printf("+-%.15s-+-%.50s-+\n", LINE_STRING, LINE_STRING);
    pShell->printf("| %-15s | %-50s |\n", "Ext. Object", "");
    pShell->printf("+-%.15s-+-%.50s-+\n", LINE_STRING, LINE_STRING);
}







void CommandManager::PrintExternalObjectNamesOnlyFooter(Shell* pShell)
{
    pShell->printf("+-%.15s-+-%.50s-+\n", LINE_STRING, LINE_STRING);

}




void CommandManager::PrintExternalObjectNamesOnlyTableDataRow(Shell* pShell, char* pObjectName, uint32_t rowIndex)
{
    if (rowIndex == 0)
    {
        pShell->printf("| %-15s | %-50s |\n", pObjectName, "Type \"<objname> help\" for help on that object");
    }
    else
    {
        pShell->printf("| %-15s | %-50s |\n", pObjectName, "");
    }



}






void CommandManager::PrintExternalObjectNamesOnly(Shell* pShell)
{
    uint32_t numObjectCommandTables;
    bool bOKToContinue = true;
    ObjectCommandTableDescriptor* pDescriptor;


    GetNumObjects(&numObjectCommandTables);

    if (numObjectCommandTables > 0)
    {
        PrintExternalObjectNamesOnlyHeader(pShell);

        for (uint32_t i = 0; i < numObjectCommandTables; i++)
        {
            bOKToContinue = GetObjectDescriptor(i, &pDescriptor);

            if (bOKToContinue)
            {
                PrintExternalObjectNamesOnlyTableDataRow(pShell, (char*)pDescriptor->pObjectName, i);
            }
        }

        PrintExternalObjectNamesOnlyFooter(pShell);
    }
}




void CommandManager::ResetCompletion(void)
{
    m_bLayer1CompletionDoObjectLookup = true;
    m_nextCompletionStartIndex = 0;
}






char* CommandManager::GetNextCompletionCandidate(int argc, char* argv[], int argToComplete)
{
    char* candidate = nullptr;





    if (argToComplete == 0)
    {
        if (argc == 0)
        {

            candidate = GetNextCompletionCandidateLayer1((char*)"");
        }
        else
        {

            candidate = GetNextCompletionCandidateLayer1(argv[argToComplete]);
        }
    }
    else if (argToComplete == 1)
    {
        if (argc == 1)
        {

            candidate = GetNextCompletionCandidateLayer2(argv[0], (char*)"");
        }
        else
        {

            candidate = GetNextCompletionCandidateLayer2(argv[0], argv[argToComplete]);
        }
    }

    return candidate;
}



char* CommandManager::GetNextCompletionCandidateLayer1(char* token)
{
    char* candidate = nullptr;







    if (m_bLayer1CompletionDoObjectLookup == true)
    {
        candidate = SearchForPartialObjectName(m_nextCompletionStartIndex, token);


        if (candidate == nullptr)
        {
            m_bLayer1CompletionDoObjectLookup = false;
            m_nextCompletionStartIndex = 0;
        }
    }



    if (m_bLayer1CompletionDoObjectLookup == false)
    {
        candidate = SearchForPartialBuiltInCommand(m_nextCompletionStartIndex, token);


        if (candidate == nullptr)
        {
            m_bLayer1CompletionDoObjectLookup = true;
            m_nextCompletionStartIndex = 0;
        }
    }






    return candidate;
}







char* CommandManager::GetNextCompletionCandidateLayer2(char* objectName, char* token)
{
    char* candidate = nullptr;





    candidate = SearchForPartialObjectCommand(objectName, m_nextCompletionStartIndex, token);

    if (candidate == nullptr)
    {
        m_nextCompletionStartIndex = 0;
    }

    return candidate;
}






char* CommandManager::SearchForPartialObjectName(uint32_t startIndex, char* token)
{
    char* candidate = nullptr;
    ObjectCommandTableDescriptor* pDescriptor;

    for (uint32_t i = startIndex; i < m_numObjectCommandTables; i++)
    {
        pDescriptor = &m_objectCommandTableDescriptors[i];

        if (strlen(pDescriptor->pObjectName) > 0)
        {
            if (strncmp(pDescriptor->pObjectName, token, strlen(token)) == 0)
            {
                candidate = (char*)pDescriptor->pObjectName;

                m_nextCompletionStartIndex = i + 1;

                break;
            }
        }
    }

    return candidate;
}




char* CommandManager::SearchForPartialObjectCommand(char* objectName, uint32_t startIndex, char* token)
{
    char* candidate = nullptr;
    uint32_t tokenLength;
    bool bFoundDescriptor;
    ObjectCommandTableDescriptor* pDescriptor;

    bFoundDescriptor = GetObjectDescriptor(objectName, &pDescriptor);

    if (bFoundDescriptor)
    {
        tokenLength = strlen(token);

        for (uint32_t i = startIndex; i < pDescriptor->tableLength; i++)
        {
            if (strlen(pDescriptor->pTable[i].commandName) > 0)
            {
                if (strncmp(pDescriptor->pTable[i].commandName, token, tokenLength) == 0)
                {
                    candidate = (char*)pDescriptor->pTable[i].commandName;

                    m_nextCompletionStartIndex = i + 1;

                    break;
                }
            }
        }
    }


    return candidate;
}





char* CommandManager::SearchForPartialBuiltInCommand(uint32_t startIndex, char* token)
{
    char* candidate = nullptr;
    ObjectCommandTableDescriptor* pDescriptor = &m_builtInCommandTableDescriptor;
    uint32_t tokenLength;

    tokenLength = strlen(token);

    for (uint32_t i = startIndex; i < pDescriptor->tableLength; i++)
    {
        if (strlen(pDescriptor->pTable[i].commandName) > 0)
        {
            if (strncmp(pDescriptor->pTable[i].commandName, token, tokenLength) == 0)
            {
                candidate = (char*)pDescriptor->pTable[i].commandName;

                m_nextCompletionStartIndex = i + 1;

                break;
            }
        }
    }

    return candidate;
}






