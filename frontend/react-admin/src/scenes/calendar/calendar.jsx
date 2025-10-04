import { useState } from "react";
import FullCalendar from "@fullcalendar/react";
import { formatDate } from "@fullcalendar/core";
import dayGridPlugin from "@fullcalendar/daygrid";
import timeGridPlugin from "@fullcalendar/timegrid";
import interactionPlugin from "@fullcalendar/interaction";
import listPlugin from "@fullcalendar/list";
import {
  Box,
  Typography,
  useTheme,
} from "@mui/material";
import Header from "../../components/Header";
import { tokens } from "../../theme";

const Calendar = () => {
  const theme = useTheme();
  const colors = tokens(theme.palette.mode);
  const [currentEvents, setCurrentEvents] = useState([]);

  const handleDateClick = (selected) => {
    const title = prompt("Please enter a new title for your event");
    const calendarApi = selected.view.calendar;
    calendarApi.unselect();

    if (title) {
      calendarApi.addEvent({
        id: `${selected.dateStr}-${title}`,
        title,
        start: selected.startStr,
        end: selected.endStr,
        allDay: selected.allDay,
      });
    }
  };

  const handleEventClick = (selected) => {
    if (
      window.confirm(
        `Are you sure you want to delete the event '${selected.event.title}'`
      )
    ) {
      selected.event.remove();
    }
  };

  return (
    <Box m="20px">
      <Header title="Academic Calendar" subtitle="Academic Seasons & Exam Weeks" />

      {/* Academic Season Legend */}
      <Box mb="20px" display="flex" gap="20px" flexWrap="wrap">
        <Box display="flex" alignItems="center" gap="8px">
          <Box width="20px" height="20px" bgcolor="#4CAF50" borderRadius="3px" />
          <Typography>Prelim Season</Typography>
        </Box>
        <Box display="flex" alignItems="center" gap="8px">
          <Box width="20px" height="20px" bgcolor="#2196F3" borderRadius="3px" />
          <Typography>Midterm Season</Typography>
        </Box>
        <Box display="flex" alignItems="center" gap="8px">
          <Box width="20px" height="20px" bgcolor="#FF9800" borderRadius="3px" />
          <Typography>Finals Season</Typography>
        </Box>
        <Box display="flex" alignItems="center" gap="8px">
          <Box width="20px" height="20px" bgcolor="#f44336" borderRadius="3px" />
          <Typography>Exam Weeks</Typography>
        </Box>
      </Box>

      {/* FULL WIDTH CALENDAR */}
      <Box width="100%">
        <FullCalendar
          height="75vh"
          plugins={[
            dayGridPlugin,
            timeGridPlugin,
            interactionPlugin,
            listPlugin,
          ]}
          headerToolbar={{
            left: "prev,next today",
            center: "title",
            right: "dayGridMonth,timeGridWeek,timeGridDay,listMonth",
          }}
          initialView="dayGridMonth"
          editable={true}
          selectable={true}
          selectMirror={true}
          dayMaxEvents={true}
          select={handleDateClick}
          eventClick={handleEventClick}
          eventsSet={(events) => setCurrentEvents(events)}
          initialEvents={[
            // PRELIM SEASON (6 weeks)
            {
              id: "prelim-season",
              title: "Prelim Season",
              start: "2025-09-01",
              end: "2025-10-12", // 6 weeks
              display: "background",
              color: "#4CAF50",
              opacity: 0.3,
            },
            // PRELIM EXAM WEEK (Week 7)
            {
              id: "prelim-exam-week",
              title: "Prelim Exam Week",
              start: "2025-10-13",
              end: "2025-10-19",
              display: "background",
              color: "#f44336",
              opacity: 0.6,
            },
            
            // MIDTERM SEASON (6 weeks)
            {
              id: "midterm-season",
              title: "Midterm Season",
              start: "2025-10-20",
              end: "2025-11-30", // 6 weeks
              display: "background",
              color: "#2196F3",
              opacity: 0.3,
            },
            // MIDTERM EXAM WEEK (Week 7)
            {
              id: "midterm-exam-week",
              title: "Midterm Exam Week",
              start: "2025-12-01",
              end: "2025-12-07",
              display: "background",
              color: "#f44336",
              opacity: 0.6,
            },
            
            // FINALS SEASON (6 weeks)
            {
              id: "finals-season",
              title: "Finals Season",
              start: "2025-12-08",
              end: "2026-01-18", // 6 weeks
              display: "background",
              color: "#FF9800",
              opacity: 0.3,
            },
            // FINALS EXAM WEEK (Week 7)
            {
              id: "finals-exam-week",
              title: "Finals Exam Week",
              start: "2026-01-19",
              end: "2026-01-25",
              display: "background",
              color: "#f44336",
              opacity: 0.6,
            },
          ]}
          eventDisplay="block"
          dayMaxEventRows={false}
          moreLinkClick="popover"
        />
      </Box>
    </Box>
  );
};

export default Calendar;