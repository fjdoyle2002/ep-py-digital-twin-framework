# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 11:27:38 2023

@author: doylef
"""
import sys 
import os
import pandas as pd
import configparser
import datetime
from retrieval.core_retrieval import CoreRetrieval
from opcmodule.opcmodule import OPCUAModule
#import conversion

from dateutil.parser import parse
import logging
from persistence.postgres_persistence_etv import PostgresPersistenceETV as PostgresPersistence
logger = logging.getLogger(__name__)



class DigitalTwin:
    def __init__(self, working_directory, start_dt_str):
        """
        function defines four global objects that contain required data for execution
        throughout rest of program code
        1) config - a configuration object read from a .ini style file
        2) signals_df - a pandas dataframe with information regarding signals to be 
        retrieved via some external source (e.g. Seeq)
        3) sensors_df - a dataframe of sensor data that is to be retrieved and persisted as 
        the embodiment of the digital twin representation
        4) actuators_df - a dataframe defining the actuators to be overriden in EP
        
        Returns
        -------
        None.
    
        """
        self.working_directory = working_directory
        
        #first get configuration file, which has info about everything else
        #directory with config file is the only required invocation argument for running this program
        config_filename = "config"
        config_suffix = "ini"
        config_path = os.path.join(working_directory, '.'.join((config_filename, config_suffix)))
        
        self.config = configparser.ConfigParser()
        self.config.sections()
        self.config.read(config_path)
        
        #put correct EnergyPlus directory in path prior to invocation of EP manager class 
        #and associated imports
        sys.path.insert(0, self.config.get('ENERGYPLUS', 'EnergyPlusDirectory'))
        #"signals" are the physical, real world data we will feed into virtual twin
        signals_path = os.path.join(working_directory, self.config.get('CONFIGURATIONFILES', 'SignalsFile'))
        self.signals_df = pd.read_csv(signals_path)
        self.signals_df = self.signals_df.assign(current_val=-1)
        #"sensors" are the energyplus variables we want to collect and persist as the representation of the digital twin
        #we include both variables termed as "sensors" in EP as well as "meters"
        sensors_path = os.path.join(working_directory, self.config.get('CONFIGURATIONFILES', 'SensorsFile'))
        self.sensors_df = pd.read_csv(sensors_path)
        self.sensors_df = self.sensors_df.assign(ep_handle=-1)
        self.sensors_df = self.sensors_df.assign(current_val=-1)
        #actuators are the settings in energyplus simulation we will be overriding
        actuators_path = os.path.join(working_directory, self.config.get('CONFIGURATIONFILES', 'ActuatorsFile'))
        self.actuators_df = pd.read_csv(actuators_path)
        self.actuators_df = self.actuators_df.assign(ep_handle=-1)
        self.actuators_df = self.actuators_df.assign(current_val=-1)
        #custom are the optional reflection based fundtions that the user can define to be called via reflection
        #at specified timepoints
        custom_path = os.path.join(working_directory, self.config.get('CONFIGURATIONFILES', 'CustomFile'))
        self.custom_callbacks_df = pd.read_csv(custom_path)
        #print(self.custom_callbacks_df)
       
        #todo create based on reflection and config spec
        self.persistence_agent = PostgresPersistence(self.config, self.sensors_df)
        # Load existing units from database into sensors_df
        # This happens BEFORE RDD parsing, so RDD can still update NULL units
        self.persistence_agent.load_units_into_sensors_df()


        #retrieval agent is the object we use to obtain real world signals
        self.retrieval_agent = CoreRetrieval(self.config, self.signals_df)
        #handle OPC UA if active
        self.opc_module = None
        if self.config.get('OPCSERVER', 'OpcServerEnabled').lower() == 'true':
            
            self.opc_module = OPCUAModule(working_directory, self.config)
            self.opc_module.start()  
            self.retrieval_agent.add_retrieval_agent(self.opc_module) 

    
        #determine start and end date of simulation
        self.run_length = int(self.config.get('DEFAULT', 'RunLength'))
        self.warmup = int(self.config.get('DEFAULT', 'WarmUpPeriodInDays'))
        self.start_dt = datetime.datetime.now()
        if start_dt_str is not None:
            try:
               self.start_dt = parse(start_dt_str)
            except:
                print("Invalid date time provided as argument:{}, exiting...".format(start_dt_str))
                sys.exit(1)
        #date we want data collection to start may differ based on warmup period
        self.start_date_for_data = self.start_dt
        #if warmup period requested, adjust start date
        if self.warmup != 0 :
            self.start_dt = self.start_dt - datetime.timedelta(days=self.warmup)
        print("Requested start date: "+str(self.start_date_for_data))
        print("Simulation start date (with warmup): "+str(self.start_dt))

        self.end_dt = self.start_date_for_data + datetime.timedelta(days=self.run_length)
        self.start_year = self.start_dt.year
        begin_year = self.start_dt.strftime('%Y')
        end_year = self.end_dt.strftime('%Y')
        begin_month = self.start_dt.strftime('%m')
        end_month = self.end_dt.strftime('%m')
        begin_day = self.start_dt.strftime('%d')
        end_day = self.end_dt.strftime('%d')
        begin_day_name = self.start_dt.strftime('%A')
        print('begin_year:{}'.format(begin_year))
        print('begin_month:{}'.format(begin_month))
        print('begin_day:{}'.format(begin_day))
        print('begin_day_name:{}'.format(begin_day_name))

            
        self.override_map = {'!- Begin Year': "  {},                                   !- Begin Year\n".format(begin_year),
                            '!- Begin Month': "  {},                                      !- Begin Month\n".format(begin_month),
                             '!- Begin Day of Month': "  {},                                      !- Begin Day of Month\n".format(begin_day),
                             '!- End Year': "  {},                                   !- End Year\n".format(end_year),
                             '!- End Month': "  {},                                     !- End Month\n".format(end_month),
                             '!- End Day of Month': "  {},                                     !- End Day of Month\n".format(end_day),
                             '!- Day of Week for Start Day': "  {},                              !- Day of Week for Start Day\n".format(begin_day_name)
                             }
    '''
    def store_simulated_signals(self, timestamp):
        output_line = timestamp.strftime("%m/%d/%Y, %H:%M:%S")
        for idx in self.sensors_df.index:
            output_line += (',{}'.format(self.sensors_df['current_val'][idx]))
        output_line += ('\n')
        
        with open('test_output.csv', 'a') as test_output_file:
            test_output_file.write(output_line)
        
        
    def store_actuator_vals(self, timestamp):
        output_line = timestamp.strftime("%m/%d/%Y, %H:%M:%S")
        for idx in self.actuators_df.index:
            output_line += (',{}'.format(self.actuators_df['current_val'][idx]))
        output_line += ('\n')
        
        with open('test_actuator_output.csv', 'a') as test_actuator_output_file:
            test_actuator_output_file.write(output_line)
            '''
    def store_simulated_signals(self, timestamp):
        if timestamp >= self.start_date_for_data:
            self.persistence_agent.persist(timestamp)
            #this is also the appropriate time to publish OPC signals if enabled
            if self.opc_module is not None:
                self.opc_module.update_variables(self.sensors_df)
            
    def get_signals_for_timepoint(self, timepoint):
       self.retrieval_agent.retrieve_signals_for_actuators_at_timepoint(self.signals_df, timepoint)
                                                                        
              
"""
Entry point for code execution
"""
    
if __name__ == "__main__":
    #the only arguments needed are the working directory and start datetime, other settings
    #will be loaded from config file that should be present there
    logging.basicConfig(filename='ep_digital_twin.log', level=logging.INFO)
    logger.info('Starting digital twin...')
    working_directory = sys.argv[1];
    start_datetime_string = sys.argv[2] if len(sys.argv) > 2 else datetime.datetime.now().strftime('%Y-%m-%d')
    
    dt = DigitalTwin(working_directory, start_datetime_string)
    
    #import statement delayed to allow for dynamically specified
    #version of EnergyPlus based on config file
    import simulator.ep_manager as epm 
    epmgr = epm.EpManager(dt)
    epmgr.invoke_simulation()
        

    
    
    
    
    